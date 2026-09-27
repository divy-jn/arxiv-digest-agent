from app.qa.citations import format_citations
from app.qa.evidence import (
    COMBINED_THRESHOLD,
    evidence_score, is_strong_evidence,
)


def test_citations_are_human_friendly_and_deduplicated():
    chunks = [{"metadata": {"section": "Methodology", "page_start": 5}}, {"metadata": {"section": "Methodology", "page_start": 5}}]
    assert format_citations(chunks) == ["Methodology, p. 5"]


def test_strong_evidence_accepted():
    """Chunks with low cosine distance AND keyword overlap should score above threshold."""
    relevant = [
        {"text": "The method compresses the key value cache with quantization.", "distance": 0.2, "metadata": {}},
    ]
    score = evidence_score("How does the method compress the cache?", relevant)
    assert score >= COMBINED_THRESHOLD
    assert is_strong_evidence("How does the method compress the cache?", relevant) is True


def test_moderate_distance_with_coverage_still_passes():
    """Distance 0.7 is poor, but high keyword overlap keeps it alive under the soft gate."""
    moderate = [
        {"text": "The method compresses the key value cache with quantization.", "distance": 0.7, "metadata": {}},
    ]
    # User's gate: only reject when distance >= 0.70 AND coverage < 0.1.
    # Here coverage is high (~0.6), so the hard gate does NOT fire.
    score = evidence_score("How does the method compress the cache?", moderate)
    assert score > 0.0  # soft gate passes because coverage is strong


def test_empty_chunks_rejected():
    """No chunks → score 0.0 and strong evidence is False."""
    assert evidence_score("What is the weather on Mars?", []) == 0.0
    assert is_strong_evidence("What is the weather on Mars?", []) is False


def test_truly_off_topic_rejected():
    """High distance AND zero keyword overlap → hard gate fires, score 0.0."""
    off_topic = [
        {"text": "This paper proposes a novel architecture for graph neural networks.", "distance": 0.75, "metadata": {}},
    ]
    score = evidence_score("What is the weather in Tokyo?", off_topic)
    assert score == 0.0
    assert is_strong_evidence("What is the weather in Tokyo?", off_topic) is False


