from datetime import datetime, timezone

from app.arxiv.models import Paper
from app.pdf.chunker import chunk_blocks
from app.pdf.extractor import ExtractedBlock
from app.qa.answer import REFUSAL
from app.qa.citations import format_citations
from app.qa.evidence import is_strong_evidence


def test_parsing_to_chunking_to_grounded_evidence():
    paper = Paper("2401.12345", "Cache Paper", ["A"], "summary", datetime.now(timezone.utc), None, [], "https://arxiv.org/abs/2401.12345", "")
    chunks = chunk_blocks([ExtractedBlock("We compress the KV cache using group quantization. Experiments report lower memory use.", 3, "Method")], paper.paper_id, paper.arxiv_id, target_tokens=100)
    retrieved = [{"text": chunks[0].text, "metadata": {"paper_id": paper.paper_id, "section": chunks[0].section, "page_start": chunks[0].page_start}, "distance": 0.1}]
    assert is_strong_evidence("How is the KV cache compressed?", retrieved)
    assert format_citations(retrieved) == ["Method, p. 3"]


def test_unsupported_question_uses_refusal_contract():
    assert "sufficient evidence" in REFUSAL
