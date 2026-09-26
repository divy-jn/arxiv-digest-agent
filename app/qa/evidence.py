from __future__ import annotations

import re


def evidence_score(question: str, chunks: list[dict[str, object]]) -> float:
    if not chunks:
        return 0.0
    stopwords = {"what", "who", "where", "when", "why", "how", "the", "this", "that", "paper", "does", "did", "is", "are", "and"}
    terms = {term for term in re.findall(r"[a-z0-9]{3,}", question.lower()) if term not in stopwords}
    text = " ".join(str(chunk["text"]).lower() for chunk in chunks)
    coverage = len(terms & set(re.findall(r"[a-z0-9]{3,}", text))) / max(1, len(terms)) if terms else 0.0
    best_distance = min(float(chunk.get("distance", 1.0)) for chunk in chunks)
    # Root cause fix: Prioritize semantic similarity (1 - distance) over exact lexical coverage (which fails on conceptual questions)
    score = min(1.0, coverage * 0.2 + max(0.0, 1 - best_distance) * 0.8)
    print(f"[DEBUG] terms: {terms}, coverage: {coverage:.2f}, best_dist: {best_distance:.2f}, score: {score:.2f}")
    return score


def is_strong_evidence(question: str, chunks: list[dict[str, object]]) -> bool:
    return bool(chunks) and evidence_score(question, chunks) >= 0.15
