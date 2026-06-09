import os
import pickle
import logging
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from openai import OpenAI

# Import from ingestion (BM25Encoder and EmbeddingGenerator)
from src.ingestion import BM25Encoder, EmbeddingGenerator

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Query Rewriter
# -------------------------------------------------------------------------
def rewrite_query(
    query: str, chat_history: List[Dict[str, str]], use_openai: bool = False
) -> str:
    if not chat_history:
        return query.strip()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if use_openai and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"""Given the following conversation history and the latest user query, 
rewrite the latest query into a standalone, specific search query that captures the user's intent.
Do not answer the query. Just output the rewritten search query.

Conversation History:
{chat_history}

Latest Query: {query}

Rewritten search query:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            rewritten = response.choices[0].message.content.strip()
            print(f"Query Rewriting (OpenAI): '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            print(
                f"Error in OpenAI query rewriting: {e}. Falling back to offline rewriter."
            )

    # Local heuristic rewriter
    history_terms = []
    tech_keywords = {
        "rate",
        "limit",
        "limiter",
        "gateway",
        "auth",
        "token",
        "redis",
        "jwt",
        "scope",
        "authentication",
        "authorization",
        "routing",
        "istio",
        "mesh",
        "circuit",
        "breaker",
        "outlier",
        "ejection",
        "oauth",
        "oauth2",
        "api",
        "proxy",
        "load",
        "balance",
        "health",
        "check",
        "timeout",
        "retry",
    }

    for msg in chat_history[-2:]:
        content = msg.get("content", "")
        words = [word.strip(",.?!()\"'") for word in content.split()]
        tech_terms = [w for w in words if w.lower() in tech_keywords]
        history_terms.extend(tech_terms)

    unique_terms = list(dict.fromkeys(history_terms))
    if unique_terms:
        rewritten = f"{' '.join(unique_terms)} {query}"
        print(f"Query Rewriting (Local Heuristic): '{query}' -> '{rewritten}'")
        return rewritten

    return query.strip()


# -------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# -------------------------------------------------------------------------
def reciprocal_rank_fusion(
    dense_results: List[Any], sparse_results: List[Any], k: int = 60, top_k: int = 20
) -> List[Dict[str, Any]]:
    scores = {}
    payloads = {}

    for rank, result in enumerate(dense_results):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank + 1)
        payloads[result.id] = result.payload

    for rank, result in enumerate(sparse_results):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (k + rank + 1)
        payloads[result.id] = result.payload

    sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    fused_results = []
    for idx, (point_id, fused_score) in enumerate(sorted_ids):
        normalized_score = max(0.0, min(1.0, fused_score))
        fused_results.append(
            {
                "id": point_id,
                "score": round(normalized_score, 4),
                "rank": idx + 1,
                "payload": payloads[point_id],
            }
        )

    return fused_results


# -------------------------------------------------------------------------
# Hybrid Retriever
# -------------------------------------------------------------------------
class HybridRetriever:
    def __init__(
        self,
        db_path: str = "data/qdrant_db",
        collection_name: str = "tech_docs",
        use_openai: bool = False,
        client: QdrantClient = None,
        bm25_encoder_path: str = None,  # NEW: explicit path override
    ):
        self.collection_name = collection_name
        self.use_openai = use_openai

        # Qdrant client
        if client is not None:
            self.client = client
            logger.info("Using shared Qdrant client.")
        else:
            logger.info(f"Connecting to Qdrant client at {db_path}...")
            self.client = QdrantClient(path=db_path)

        # BM25 Encoder path resolution (FIXED)
        if bm25_encoder_path is not None:
            encoder_path = bm25_encoder_path
        else:
            # Resolve relative to db_path's parent directory
            db_dir = os.path.dirname(os.path.abspath(db_path))
            encoder_path = os.path.join(db_dir, "bm25_encoder.pkl")

        logger.info(f"Looking for BM25 encoder at: {encoder_path}")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"BM25 Encoder pickle file not found at {encoder_path}. "
                f"Run ingestion first. (db_path was: {db_path})"
            )

        with open(encoder_path, "rb") as f:
            self.bm25_encoder = pickle.load(f)
        logger.info("BM25 Encoder loaded successfully.")

        # Embedding generator
        self.embedding_gen = EmbeddingGenerator(use_openai=use_openai)

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: dense + sparse with RRF fusion.
        Returns list of dicts with keys: id, score, rank, payload.
        """
        # DEBUG: Print query
        print("=" * 60)
        print(f"[RETRIEVE] QUERY: '{query}'")

        # Dense vector
        query_dense = self.embedding_gen.get_embeddings([query])[0]
        logger.info(f"Dense vector shape: {len(query_dense)}")

        # Sparse vector
        sparse_indices, sparse_values = self.bm25_encoder.encode(query)
        logger.info(
            f"Sparse vector: {len(sparse_indices)} indices, {len(sparse_values)} values"
        )

        # --- Dense Search ---
        dense_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_dense,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        dense_results = dense_response.points
        print(f"[DENSE] Retrieved {len(dense_results)} results")

        # --- Sparse Search ---
        sparse_results = []
        if sparse_indices:
            sparse_response = self.client.query_points(
                collection_name=self.collection_name,
                query=SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=top_k,
                with_payload=True,
            )
            sparse_results = sparse_response.points
            print(f"[SPARSE] Retrieved {len(sparse_results)} results")
        else:
            print(f"[SPARSE] No sparse indices generated (empty query?)")

        # --- RRF Fusion ---
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=top_k)
        print(f"[RRF] Fused into {len(fused)} results")

        # DEBUG: Print top results
        for i, item in enumerate(fused[:5]):
            text = item.get("payload", {}).get("text", "NO TEXT")
            print(f"  [{i}] id={item['id']} score={item['score']} | {text[:100]}...")

        print("=" * 60)

        return fused
