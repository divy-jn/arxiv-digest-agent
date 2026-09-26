from app.qa.citations import format_citations
from app.qa.evidence import evidence_score, is_strong_evidence


def test_citations_are_human_friendly_and_deduplicated():
    chunks = [{"metadata": {"section": "Methodology", "page_start": 5}}, {"metadata": {"section": "Methodology", "page_start": 5}}]
    assert format_citations(chunks) == ["Methodology, p. 5"]


def test_evidence_gate_accepts_relevant_and_rejects_empty():
    relevant = [{"text": "The method compresses the key value cache with quantization.", "distance": 0.1, "metadata": {}}]
    assert evidence_score("How does the method compress the cache?", relevant) > 0.32
    assert is_strong_evidence("What is the weather on Mars?", []) is False
