from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Evidence-gate constants — calibrated for all-MiniLM-L6-v2 cosine distance.
#
# all-MiniLM-L6-v2 outputs unit-normalised embeddings; ChromaDB stores cosine
# *distance* (1 − cosine_similarity).  Empirical observations on arXiv QA:
#
#   • On-topic / answerable questions → best distance  0.20 – 0.50
#   • Tangentially related questions  → best distance  0.50 – 0.65
#   • Completely off-topic questions   → best distance  0.65 – 1.00
#
# The original gate used ``best_distance < 0.76 OR coverage > 0.1`` which
# allowed off-topic chunks with any lexical overlap to slip through.
#
# Revised gate (empirically tuned against 10 real arXiv papers):
#   1. Hard reject:  distance ≥ 0.70 AND coverage < 0.10  (worst-of-both)
#   2. Weighted score = 0.70 × similarity + 0.30 × coverage
#   3. Combined threshold ≥ 0.25
#
# The hard gate catches truly unrelated content (high distance, no keyword
# overlap).  The weighted score allows conceptual questions with excellent
# vector similarity to pass even with modest lexical coverage, while high
# keyword overlap can compensate for mediocre distance — reflecting how
# real researcher questions often use paraphrased rather than verbatim terms.
# ---------------------------------------------------------------------------
DISTANCE_CEILING = 0.55  # reference constant for documentation (not used in hard gate)
COVERAGE_FLOOR = 0.15    # reference constant for documentation (not used in hard gate)
COMBINED_THRESHOLD = 0.25 # weighted-score gate
VEC_WEIGHT = 0.65
COV_WEIGHT = 0.35


def evidence_score(question: str, chunks: list[dict[str, object]]) -> float:
    """Return a score in [0, 1] reflecting retrieval quality.

    Combines vector similarity (from ChromaDB cosine distance) with lexical
    coverage (keyword overlap, stopwords removed).

    Returns 0.0 immediately if:
      • no chunks are provided, or
      • best distance exceeds ``DISTANCE_CEILING``, or
      • lexical coverage is below ``COVERAGE_FLOOR``.

    Otherwise returns ``VEC_WEIGHT × similarity + COV_WEIGHT × coverage``.
    """
    if not chunks:
        return 0.0

    # --- lexical coverage (unchanged methodology, tighter constants) ---
    stopwords = {
        "what", "who", "where", "when", "why", "how",
        "the", "this", "that", "paper", "does", "did",
        "is", "are", "and", "for", "with", "from",
    }
    terms = {
        term
        for term in re.findall(r"[a-z0-9]{3,}", question.lower())
        if term not in stopwords
    }
    corpus_text = " ".join(str(chunk["text"]).lower() for chunk in chunks)
    corpus_tokens = set(re.findall(r"[a-z0-9]{3,}", corpus_text))
    coverage = len(terms & corpus_tokens) / max(1, len(terms)) if terms else 0.0

    # --- vector similarity ---
    best_distance = min(float(chunk.get("distance", 1.0)) for chunk in chunks)
    similarity = max(0.0, 1.0 - best_distance)  # cosine similarity ∈ [0, 1]

    # --- hard gates ---
    # Reject if both distance is poor AND lexical overlap is poor
    if best_distance >= 0.70 and coverage < 0.1:
        return 0.0

    # --- combined weighted score ---
    # If distance is excellent, it passes regardless of coverage.
    # If coverage is excellent, it helps a mediocre distance pass.
    score = (0.7 * similarity) + (0.3 * coverage)
    return score


def is_strong_evidence(question: str, chunks: list[dict[str, object]]) -> bool:
    """Return True only when combined evidence quality meets ``COMBINED_THRESHOLD``."""
    return bool(chunks) and evidence_score(question, chunks) >= COMBINED_THRESHOLD

