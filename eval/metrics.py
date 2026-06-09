from typing import List, Dict, Any, Union, Set


def evaluate_retrieval(
    query: str,
    ground_truth_chunks: Union[List[str], str, None],
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Computes retrieval evaluation metrics.

    ground_truth_chunks: Either:
      - List of expected chunk IDs (strings), OR
      - A text string (will attempt fuzzy matching against payloads)
    retrieved_chunks: List of retrieved chunk dicts with 'id' or payload['chunk_id']
    """
    # Extract retrieved IDs
    retrieved_ids = []
    for c in retrieved_chunks:
        payload = c.get("payload", {})
        if "chunk_id" in payload:
            retrieved_ids.append(str(payload["chunk_id"]))
        elif "id" in c:
            retrieved_ids.append(str(c["id"]))
        else:
            retrieved_ids.append(str(c))  # fallback

    retrieved_set = set(retrieved_ids)

    # Handle ground_truth format
    if ground_truth_chunks is None:
        gt_set = set()
    elif isinstance(ground_truth_chunks, list):
        gt_set = set(str(x) for x in ground_truth_chunks)
    elif isinstance(ground_truth_chunks, str):
        # If it's text, we can't do ID-based matching
        # Return metrics based on text overlap (fallback)
        gt_text = ground_truth_chunks.lower().strip()
        gt_set = set()  # empty ID set means we use text fallback below
    else:
        gt_set = set(str(ground_truth_chunks))

    # If no valid ID-based ground truth, use text overlap fallback
    if not gt_set and isinstance(ground_truth_chunks, str):
        gt_text = ground_truth_chunks.lower().strip()
        hits = 0
        for idx, cid in enumerate(retrieved_ids):
            c = retrieved_chunks[idx] if idx < len(retrieved_chunks) else {}
            payload = c.get("payload", c)
            chunk_text = str(payload.get("text", payload.get("content", ""))).lower()
            # Consider a hit if chunk contains a significant portion of ground truth
            if len(gt_text) > 20 and gt_text[:50] in chunk_text:
                hits += 1
            elif gt_text in chunk_text:
                hits += 1

        precision = hits / len(retrieved_set) if retrieved_set else 0.0
        recall = (
            1.0 if hits > 0 else 0.0
        )  # binary: did we retrieve at least one relevant chunk?
        mrr = 1.0 if hits > 0 and retrieved_ids else 0.0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            "retrieval_precision": float(precision),
            "retrieval_recall": float(recall),
            "retrieval_f1": float(f1),
            "mrr": float(mrr),
        }

    # Standard ID-based matching
    if not gt_set:
        return {
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "retrieval_f1": 0.0,
            "mrr": 0.0,
        }

    hits = len(retrieved_set & gt_set)

    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    recall = hits / len(gt_set) if gt_set else 0.0

    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, cid in enumerate(retrieved_ids):
        if cid in gt_set:
            mrr = 1.0 / (idx + 1)
            break

    return {
        "retrieval_precision": float(precision),
        "retrieval_recall": float(recall),
        "retrieval_f1": float(f1),
        "mrr": float(mrr),
    }
