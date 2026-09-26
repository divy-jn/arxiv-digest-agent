import pytest

from app.exceptions import ParseError
from app.pdf.chunker import chunk_blocks
from app.pdf.extractor import ExtractedBlock, validate_extraction
from app.pdf.sections import detect_section


def test_detect_major_sections():
    assert detect_section("1 Introduction") == "Introduction"
    assert detect_section("REFERENCES") == "References"
    assert detect_section("normal paragraph sentence.") is None


def test_chunking_keeps_metadata_and_sections():
    blocks = [ExtractedBlock("alpha " * 200, 1, "Introduction"), ExtractedBlock("beta " * 200, 2, "Method")]
    chunks = chunk_blocks(blocks, "p1", "2401.12345", target_tokens=50, overlap_tokens=5)
    assert chunks[0].paper_id == "p1"
    assert chunks[0].section == "Introduction"
    assert chunks[1].section == "Method"
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)


def test_validation_rejects_empty_extract():
    with pytest.raises(ParseError):
        validate_extraction([])
