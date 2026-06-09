from typing import List, Dict, Any, Tuple


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False

        try:
            from sentence_transformers import CrossEncoder

            print(f"Loading CrossEncoder '{self.model_name}'...")
            # Load cross-encoder model
            self.model = CrossEncoder(self.model_name)
            self.is_loaded = True
            print("CrossEncoder loaded successfully.")
        except Exception as e:
            print(
                f"WARNING: Could not load CrossEncoder model ({e}). Reranker will fall back to RRF scores."
            )

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reranks candidate chunks based on query-chunk semantic match using a CrossEncoder.
        Returns top_n items as Tuples of (CandidateDict, ScoreFloat).
        """
        if not candidates:
            return []

        if not self.is_loaded:
            # Fallback: Sort by their retrieval RRF score
            sorted_candidates = sorted(
                candidates, key=lambda x: x["score"], reverse=True
            )[:top_n]
            return [(cand, cand["score"]) for cand in sorted_candidates]

        try:
            # Prepare pairs for cross-encoder inference: [ [query, chunk_text], ... ]
            pairs = [[query, cand["payload"]["text"]] for cand in candidates]
            scores = self.model.predict(pairs)

            # Map predictions to candidates
            scored_candidates = []
            for cand, score in zip(candidates, scores):
                scored_candidates.append((cand, float(score)))

            # Sort descending by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[:top_n]
        except Exception as e:
            print(
                f"Error during CrossEncoder reranking: {e}. Falling back to RRF scores."
            )
            sorted_candidates = sorted(
                candidates, key=lambda x: x["score"], reverse=True
            )[:top_n]
            return [(cand, cand["score"]) for cand in sorted_candidates]
