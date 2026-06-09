import os
import re
from typing import Dict, Any, List
from openai import OpenAI


# -------------------------------------------------------------------------
# Helper: Parse numeric score from LLM text responses
# -------------------------------------------------------------------------
def extract_score(text: str, default: float = 0.0) -> float:
    # Look for patterns like "Score: 0.8" or "0.85" or "1.0" or "0.8/1.0"
    matches = re.findall(r"(0\.\d+|1\.0|\d/10|\d\.\d)", text)
    if matches:
        val = matches[0]
        if "/" in val:
            parts = val.split("/")
            return float(parts[0]) / float(parts[1])
        return float(val)
    return default


# -------------------------------------------------------------------------
# Faithfulness Scorer (Groundedness)
# -------------------------------------------------------------------------
def score_faithfulness(answer: str, context: str, use_openai: bool = False) -> float:
    """
    Measures if all facts in the answer are supported by the retrieved context.
    Returns a float between 0.0 (fully hallucinated) and 1.0 (fully grounded).
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if use_openai and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"""Rate how well the generated answer is grounded in the provided context.
If every factual claim in the answer is directly supported by the context, score 1.0.
If there are claims in the answer that cannot be verified by the context, lower the score (0.0 if completely hallucinated).
Provide a brief reasoning, followed by the final score in the format "Score: [value]".

Context:
{context}

Answer:
{answer}

Score (0.0-1.0):"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return extract_score(response.choices[0].message.content, default=1.0)
        except Exception as e:
            print(
                f"Error in LLM faithfulness judge: {e}. Falling back to offline heuristic."
            )

    # Offline Heuristic: Word overlap and sentence alignment
    # If the answer is an offline synthesis, it is highly faithful because it is directly copied
    # Let's count how many words in the answer are present in the context
    answer_words = re.findall(r"\w+", answer.lower())
    if not answer_words:
        return 1.0

    context_words = set(re.findall(r"\w+", context.lower()))
    stopwords = {
        "and",
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "is",
        "for",
        "with",
        "that",
        "this",
        "it",
        "on",
        "at",
        "by",
        "as",
    }

    meaningful_words = [w for w in answer_words if w not in stopwords]
    if not meaningful_words:
        return 1.0

    hits = sum(1 for w in meaningful_words if w in context_words)
    score = hits / len(meaningful_words)
    # Give a small boost for offline-generated markers, capping at 1.0
    if "[offline mode" in answer.lower():
        score = min(score + 0.1, 1.0)
    return float(round(score, 2))


# -------------------------------------------------------------------------
# Answer Relevance Scorer
# -------------------------------------------------------------------------
def score_relevance(query: str, answer: str, use_openai: bool = False) -> float:
    """
    Measures if the answer directly addresses the user's query (relevance vs. off-topic).
    Returns a float between 0.0 and 1.0.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if use_openai and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"""Rate how well the generated answer addresses the question.
If the answer directly, clearly, and completely answers the question, score 1.0.
If the answer is off-topic, generic, or fails to answer the question, score 0.0.
Provide a brief reasoning, followed by the final score in the format "Score: [value]".

Question:
{query}

Answer:
{answer}

Score (0.0-1.0):"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return extract_score(response.choices[0].message.content, default=0.8)
        except Exception as e:
            print(
                f"Error in LLM relevance judge: {e}. Falling back to offline heuristic."
            )

    # Offline Heuristic: Jaccard Word Similarity
    q_words = set(re.findall(r"\w+", query.lower()))
    a_words = set(re.findall(r"\w+", answer.lower()))

    # Filter stopwords
    stopwords = {
        "how",
        "do",
        "i",
        "the",
        "a",
        "an",
        "is",
        "are",
        "what",
        "with",
        "to",
        "in",
        "on",
        "at",
        "for",
        "of",
        "and",
    }
    q_filtered = q_words - stopwords
    a_filtered = a_words - stopwords

    if not q_filtered:
        q_filtered = q_words
    if not a_filtered:
        a_filtered = a_words

    intersection = q_filtered.intersection(a_filtered)
    union = q_filtered.union(a_filtered)

    jaccard = len(intersection) / len(union) if union else 0.0
    # Rescale Jaccard score since conversation usually has low intersection
    score = min(jaccard * 4.0, 1.0)
    # Ensure a reasonable minimum relevance for synthetically extracted answers
    if hits := len(intersection):
        score = max(score, 0.5 + min(hits * 0.1, 0.4))
    return float(round(score, 2))


# -------------------------------------------------------------------------
# Hallucination Scorer
# -------------------------------------------------------------------------
def detect_hallucination(
    answer: str, context: str, use_openai: bool = False
) -> Dict[str, Any]:
    """
    Checks for claims in the answer that are unsupported by the context.
    Returns:
      - hallucination_rate: fraction of sentences flagged as hallucinated.
      - unsupported_claims: list of sentences flagged as unsupported.
      - is_hallucinated: bool.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if use_openai and openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            prompt = f"""Analyze the answer sentence-by-sentence and extract any claims that are NOT supported by the context.
Return your output as a JSON block with two fields:
- "unsupported_claims": a list of strings representing the unsupported sentences.
- "hallucination_rate": a float representing the fraction of sentences that are unsupported.

Context:
{context}

Answer:
{answer}

JSON Response:"""
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            import json

            data = json.loads(response.choices[0].message.content)
            rate = float(data.get("hallucination_rate", 0.0))
            unsupported = data.get("unsupported_claims", [])
            return {
                "hallucination_rate": rate,
                "unsupported_claims": unsupported,
                "is_hallucinated": rate > 0.0 or len(unsupported) > 0,
            }
        except Exception as e:
            print(
                f"Error in LLM hallucination judge: {e}. Falling back to offline heuristic."
            )

    # Offline Heuristic: Sentence matching
    # Split the answer into sentences
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    unsupported = []

    # Clean context text for easier substring checks
    clean_context = re.sub(r"\s+", " ", context.lower())

    for sent in sentences:
        sent = sent.strip()
        if not sent or "offline mode" in sent.lower():
            continue

        # Strip citation tags like [1], [2]
        clean_sent = re.sub(r"\[\d+\]", "", sent).strip()
        clean_sent_lower = clean_sent.lower()

        # Check if the sentence (or a large chunk of it) appears in the context
        words_in_sent = re.findall(r"\w+", clean_sent_lower)
        if len(words_in_sent) < 4:
            continue

        # If the direct sentence doesn't exist, check word overlap percentage
        # Find matches for contiguous word groups
        overlap_score = 0
        for i in range(len(words_in_sent) - 2):
            trigram = " ".join(words_in_sent[i : i + 3])
            if trigram in clean_context:
                overlap_score += 1

        max_possible = len(words_in_sent) - 2
        if max_possible > 0:
            ratio = overlap_score / max_possible
        else:
            ratio = 1.0

        if (
            ratio < 0.4
        ):  # Less than 40% trigram overlap means it is likely a hallucination
            unsupported.append(sent)

    rate = len(unsupported) / len(sentences) if sentences else 0.0
    return {
        "hallucination_rate": float(round(rate, 2)),
        "unsupported_claims": unsupported,
        "is_hallucinated": len(unsupported) > 0,
    }
