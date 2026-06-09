import os
import re
from typing import List, Dict, Any, Tuple
from openai import OpenAI


# -------------------------------------------------------------------------
# Context Assembler
# -------------------------------------------------------------------------
def assemble_context(
    reranked_chunks: List[Tuple[Dict[str, Any], float]],
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Deduplicates and formats reranked chunks with citation indices like [1], [2], etc.
    Returns the formatted context block and a mapping dict of citations.
    """
    context_parts = []
    citations = {}

    # Track unique texts to avoid duplicate citations
    seen_texts = set()
    citation_idx = 1

    for cand, score in reranked_chunks:
        text = cand["payload"]["text"].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)

        citation_id = f"[{citation_idx}]"
        # Ensure positive score for display
        display_score = abs(float(score))
        context_parts.append(f"{citation_id} {text}")

        citations[citation_id] = {
            "source": cand["payload"]["source"],
            "chunk_id": cand["payload"]["chunk_id"],
            "score": round(min(1.0, display_score), 4),  # Normalize to 0-1
            "text": text,
        }
        citation_idx += 1

    context_block = "\n\n".join(context_parts)  # Better formatting with newlines
    return context_block, citations


# -------------------------------------------------------------------------
# Offline Fallback Synthesizer
# -------------------------------------------------------------------------
def synthesize_offline_answer(query: str, citations: Dict[str, Dict[str, Any]]) -> str:
    """
    Heuristically extracts sentences from citations that match keywords in the query,
    assembling them into a citation-grounded response.
    """
    query_words = set(re.findall(r"\w+", query.lower()))
    # Exclude common stopwords to focus on technical terms
    stopwords = {
        "how",
        "do",
        "i",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "for",
        "to",
        "of",
        "and",
        "is",
        "are",
        "what",
        "with",
        "configured",
        "setup",
        "can",
        "you",
        "tell",
        "me",
        "about",
        "does",
        "it",
        "work",
    }
    query_keywords = query_words - stopwords
    if not query_keywords:
        query_keywords = query_words

    synthesized_sentences = []

    for citation_id, info in citations.items():
        text = info["text"]
        # Split into sentences using a simple regex
        sentences = re.split(r"(?<=[.!?])\s+", text)
        scored_sentences = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_words = set(re.findall(r"\w+", sent.lower()))
            overlap = query_keywords.intersection(sent_words)
            if overlap:
                # Score based on overlap
                score = len(overlap)
                scored_sentences.append((sent, score))

        if scored_sentences:
            # Sort sentences by match quality
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sent = scored_sentences[0][0]
            # Ensure it ends with punctuation
            if best_sent and best_sent[-1] not in ".!":
                best_sent += "."
            synthesized_sentences.append(f"{best_sent} {citation_id}")

    if not synthesized_sentences:
        # If no overlap, just grab the first sentence of each chunk
        for citation_id, info in list(citations.items())[:3]:
            sentences = re.split(r"(?<=[.!?])\s+", info["text"])
            if sentences:
                best_sent = sentences[0].strip()
                if best_sent and best_sent[-1] not in ".!":
                    best_sent += "."
                synthesized_sentences.append(f"{best_sent} {citation_id}")

    if synthesized_sentences:
        answer = " ".join(synthesized_sentences)
    else:
        answer = "I don't have enough information in the provided context to answer this query."

    # Clean up the prefix - no weird formatting
    return answer.strip()


# -------------------------------------------------------------------------
# Answer Generator
# -------------------------------------------------------------------------
def generate_answer(
    query: str,
    context: str,
    citations: Dict[str, Dict[str, Any]],
    use_openai: bool = False,
) -> Dict[str, Any]:
    """
    Generates a citation-aware answer grounded in the context.
    Calls OpenAI GPT-4o if configured, otherwise falls back to the offline synthesizer.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")

    if use_openai and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            system_prompt = """You are an expert technical assistant. Answer the user's question using ONLY 
the provided context. Every factual claim must be followed by a citation like [1], [2], etc.
Do not combine citations (e.g. [1, 2]). Use individual tags (e.g. [1] and [2]).
If the context doesn't contain the answer, say "I don't have enough information."
Do not use outside knowledge or hallucinate any facts."""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ]

            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, temperature=0.2
            )

            answer = response.choices[0].message.content.strip()

            # Extract citations that were actually printed in the answer
            used_citations = [c for c in citations.keys() if c in answer]

            return {
                "answer": answer,
                "citations_used": used_citations,
                "citation_details": {c: citations[c] for c in used_citations},
                "context_chunks": list(citations.values()),
            }
        except Exception as e:
            print(
                f"Error in OpenAI answer generation: {e}. Falling back to offline synthesizer."
            )

    # Local fallback - clean output without weird prefixes
    answer = synthesize_offline_answer(query, citations)
    used_citations = [c for c in citations.keys() if c in answer]

    return {
        "answer": answer,
        "citations_used": used_citations,
        "citation_details": {c: citations[c] for c in used_citations},
        "context_chunks": list(citations.values()),
    }
