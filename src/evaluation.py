import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from src.ingestion import EmbeddingGenerator
from src.retrieval import HybridRetriever, rewrite_query
from src.reranking import CrossEncoderReranker
from src.generation import assemble_context, generate_answer
from eval.metrics import evaluate_retrieval
from eval.judges import score_faithfulness, score_relevance, detect_hallucination

logger = logging.getLogger(__name__)


def resolve_ground_truth_ids(
    client: QdrantClient,
    collection_name: str,
    ground_truth_text: str,
    embedding_gen: EmbeddingGenerator,
    top_k: int = 5,
) -> List[str]:
    """
    Given a ground truth text string, find the chunk IDs in Qdrant that best match it.
    This bridges text-based eval datasets to ID-based metrics.
    """
    if not ground_truth_text or not isinstance(ground_truth_text, str):
        return []

    # Search for chunks similar to the ground truth text
    try:
        query_vector = embedding_gen.get_embeddings([ground_truth_text])[0]
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        ids = []
        for point in response.points:
            payload = point.payload or {}
            cid = payload.get("chunk_id", str(point.id))
            ids.append(str(cid))
        return ids
    except Exception as e:
        logger.warning(f"Could not resolve ground truth to IDs: {e}")
        return []


# -------------------------------------------------------------------------
# Naive RAG Baseline Pipeline
# -------------------------------------------------------------------------
def run_naive_rag(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embedding_gen: EmbeddingGenerator,
    use_openai: bool = False,
    top_k: int = 20,
) -> Dict[str, Any]:

    query_dense = embedding_gen.get_embeddings([query])[0]

    dense_response = client.query_points(
        collection_name=collection_name,
        query=query_dense,
        using="dense",
        limit=top_k,
        with_payload=True,
    )
    dense_results = dense_response.points

    retrieved_chunks = [
        {"id": r.id, "score": r.score, "payload": r.payload} for r in dense_results
    ]

    context, citations = assemble_context([(c, c["score"]) for c in retrieved_chunks])

    output = generate_answer(query, context, citations, use_openai=use_openai)

    return {
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": output["answer"],
        "citations_used": output["citations_used"],
        "citation_details": output["citation_details"],
    }


# -------------------------------------------------------------------------
# Production RAG Pipeline
# -------------------------------------------------------------------------
def run_production_rag(
    query: str,
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    chat_history: List[Dict[str, str]] = None,
    use_openai: bool = False,
) -> Dict[str, Any]:

    if chat_history is None:
        chat_history = []

    rewritten_query = rewrite_query(query, chat_history, use_openai=use_openai)

    candidates = retriever.retrieve(rewritten_query, top_k=20)

    reranked = reranker.rerank(rewritten_query, candidates, top_n=5)

    context, citations = assemble_context(reranked)

    output = generate_answer(query, context, citations, use_openai=use_openai)

    return {
        "rewritten_query": rewritten_query,
        "retrieved_chunks": [r[0] for r in reranked],
        "context": context,
        "answer": output["answer"],
        "citations_used": output["citations_used"],
        "citation_details": output["citation_details"],
    }


# -------------------------------------------------------------------------
# Evaluation Runner
# -------------------------------------------------------------------------
def run_full_evaluation(
    dataset_path: str = "data/eval_dataset.jsonl",
    db_path: str = "data/qdrant_db",
    collection_name: str = "tech_docs",
    output_path: str = "data/eval_results.json",
    use_openai: bool = False,
    client: QdrantClient = None,
) -> Dict[str, Any]:

    print("Loading evaluation dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    eval_dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_dataset.append(json.loads(line))

    print(f"Loaded {len(eval_dataset)} evaluation cases.")

    if client is None:
        client = QdrantClient(path=db_path)

    embedding_gen = EmbeddingGenerator(use_openai=use_openai)

    retriever = HybridRetriever(
        db_path=db_path,
        collection_name=collection_name,
        use_openai=use_openai,
        client=client,
        bm25_encoder_path=os.path.join(
            os.path.dirname(os.path.abspath(db_path)), "bm25_encoder.pkl"
        ),
    )
    reranker = CrossEncoderReranker()

    results = []

    print("Running comparative RAG evaluation pipeline...")
    for idx, item in enumerate(tqdm(eval_dataset)):
        query = item.get("query", "")
        raw_ground_truth = item.get("ground_truth_context") or item.get(
            "ground_truth", ""
        )
        category = item.get("category", "unknown")
        difficulty = item.get("difficulty", "unknown")

        # --- RESOLVE GROUND TRUTH TO CHUNK IDs ---
        if isinstance(raw_ground_truth, list):
            # Already a list of IDs
            ground_truth_ids = [str(x) for x in raw_ground_truth]
        elif isinstance(raw_ground_truth, str) and raw_ground_truth.strip():
            # Text ground truth — resolve to IDs via semantic search
            ground_truth_ids = resolve_ground_truth_ids(
                client, collection_name, raw_ground_truth, embedding_gen, top_k=3
            )
        else:
            ground_truth_ids = []

        # DEBUG
        print(f"\n{'=' * 60}")
        print(f"EVAL [{idx}] QUERY: {query}")
        print(f"GROUND TRUTH TYPE: {type(raw_ground_truth).__name__}")
        print(f"RESOLVED GT IDs: {ground_truth_ids}")

        # --- Run Naive RAG ---
        start_time = time.time()
        naive_out = run_naive_rag(
            query, client, collection_name, embedding_gen, use_openai=use_openai
        )
        naive_latency = time.time() - start_time

        print(
            f"NAIVE RETRIEVED: {[c.get('payload', {}).get('chunk_id', c['id']) for c in naive_out['retrieved_chunks'][:5]]}"
        )

        naive_retrieval_scores = evaluate_retrieval(
            query, ground_truth_ids, naive_out["retrieved_chunks"]
        )
        naive_faithfulness = score_faithfulness(
            naive_out["answer"], naive_out["context"], use_openai=use_openai
        )
        naive_relevance = score_relevance(
            query, naive_out["answer"], use_openai=use_openai
        )
        naive_hallucination = detect_hallucination(
            naive_out["answer"], naive_out["context"], use_openai=use_openai
        )

        # --- Run Production RAG ---
        start_time = time.time()
        prod_out = run_production_rag(
            query, retriever, reranker, chat_history=[], use_openai=use_openai
        )
        prod_latency = time.time() - start_time

        print(
            f"PROD RETRIEVED: {[c.get('payload', {}).get('chunk_id', c.get('id')) for c in prod_out['retrieved_chunks'][:5]]}"
        )
        print(f"NAIVE SCORES: {naive_retrieval_scores}")
        print(
            f"PROD SCORES: {evaluate_retrieval(query, ground_truth_ids, prod_out['retrieved_chunks'])}"
        )
        print(f"{'=' * 60}")

        prod_retrieval_scores = evaluate_retrieval(
            query, ground_truth_ids, prod_out["retrieved_chunks"]
        )
        prod_faithfulness = score_faithfulness(
            prod_out["answer"], prod_out["context"], use_openai=use_openai
        )
        prod_relevance = score_relevance(
            query, prod_out["answer"], use_openai=use_openai
        )
        prod_hallucination = detect_hallucination(
            prod_out["answer"], prod_out["context"], use_openai=use_openai
        )

        results.append(
            {
                "id": idx,
                "query": query,
                "category": category,
                "difficulty": difficulty,
                "naive_latency": naive_latency,
                "naive_retrieval_precision": naive_retrieval_scores[
                    "retrieval_precision"
                ],
                "naive_retrieval_recall": naive_retrieval_scores["retrieval_recall"],
                "naive_retrieval_f1": naive_retrieval_scores["retrieval_f1"],
                "naive_mrr": naive_retrieval_scores["mrr"],
                "naive_faithfulness": naive_faithfulness,
                "naive_relevance": naive_relevance,
                "naive_hallucination_rate": naive_hallucination["hallucination_rate"],
                "naive_answer": naive_out["answer"],
                "prod_latency": prod_latency,
                "prod_retrieval_precision": prod_retrieval_scores[
                    "retrieval_precision"
                ],
                "prod_retrieval_recall": prod_retrieval_scores["retrieval_recall"],
                "prod_retrieval_f1": prod_retrieval_scores["retrieval_f1"],
                "prod_mrr": prod_retrieval_scores["mrr"],
                "prod_faithfulness": prod_faithfulness,
                "prod_relevance": prod_relevance,
                "prod_hallucination_rate": prod_hallucination["hallucination_rate"],
                "prod_answer": prod_out["answer"],
            }
        )

    df = pd.DataFrame(results)

    aggregates = {
        "naive": {
            "retrieval_recall": float(df["naive_retrieval_recall"].mean()),
            "retrieval_f1": float(df["naive_retrieval_f1"].mean()),
            "mrr": float(df["naive_mrr"].mean()),
            "faithfulness": float(df["naive_faithfulness"].mean()),
            "relevance": float(df["naive_relevance"].mean()),
            "hallucination_rate": float(df["naive_hallucination_rate"].mean()),
            "latency": float(df["naive_latency"].mean()),
        },
        "production": {
            "retrieval_recall": float(df["prod_retrieval_recall"].mean()),
            "retrieval_f1": float(df["prod_retrieval_f1"].mean()),
            "mrr": float(df["prod_mrr"].mean()),
            "faithfulness": float(df["prod_faithfulness"].mean()),
            "relevance": float(df["prod_relevance"].mean()),
            "hallucination_rate": float(df["prod_hallucination_rate"].mean()),
            "latency": float(df["prod_latency"].mean()),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"aggregates": aggregates, "detailed_results": results}, f, indent=4)

    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)

    print("=" * 50)
    print("EVALUATION PIPELINE COMPLETED")
    print("=" * 50)
    print(f"Results saved to: {output_path} and {csv_path}")
    print("Metrics summary:")
    print(f"{'Metric':<25} | {'Naive (Baseline)':<18} | {'Production RAG':<18}")
    print("-" * 69)
    for m in [
        "retrieval_recall",
        "retrieval_f1",
        "mrr",
        "faithfulness",
        "relevance",
        "hallucination_rate",
        "latency",
    ]:
        n_val = aggregates["naive"][m]
        p_val = aggregates["production"][m]
        if m == "latency":
            print(
                f"{m.replace('_', ' ').title():<25} | {n_val:14.3f}s | {p_val:14.3f}s"
            )
        else:
            print(f"{m.replace('_', ' ').title():<25} | {n_val:17.3f} | {p_val:17.3f}")
    print("=" * 50)

    return aggregates


if __name__ == "__main__":
    run_full_evaluation(use_openai=False)
