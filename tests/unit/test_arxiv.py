from datetime import datetime, timezone

import pytest

from app.arxiv.client import normalize_arxiv_id, parse_arxiv_reference, parse_entries
from app.arxiv.models import Paper
from app.arxiv.search import rank_candidates


@pytest.mark.parametrize(("value", "expected"), [
    ("2603.13942", {"arxiv_id": "2603.13942", "version": None, "input_kind": "id"}),
    ("2603.13942v3", {"arxiv_id": "2603.13942", "version": "v3", "input_kind": "id"}),
    ("https://arxiv.org/abs/2603.13942v3", {"arxiv_id": "2603.13942", "version": "v3", "input_kind": "abs_url"}),
    ("https://arxiv.org/pdf/2603.13942", {"arxiv_id": "2603.13942", "version": None, "input_kind": "pdf_url"}),
    ("https://arxiv.org/html/2603.13942v3", {"arxiv_id": "2603.13942", "version": "v3", "input_kind": "html_url"}),
    ("not an id", None),
])
def test_parse_arxiv_reference(value, expected):
    assert parse_arxiv_reference(value) == expected

def test_normalize_arxiv_id():
    assert normalize_arxiv_id("https://arxiv.org/html/2603.13942v3") == "2603.13942"

def test_parse_atom_metadata():
    xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2401.12345v1</id><title> A title </title><summary> An abstract </summary><published>2024-01-02T00:00:00Z</published><updated>2024-01-03T00:00:00Z</updated><author><name>Ada</name></author><category term="cs.CL"/><link title="pdf" href="https://arxiv.org/pdf/2401.12345v1"/></entry></feed>'''
    paper = parse_entries(xml)[0]
    assert paper.arxiv_id == "2401.12345"
    assert paper.authors == ["Ada"]
    assert paper.pdf_url.endswith("2401.12345v1")
    assert paper.html_url.endswith("2401.12345v1")


def test_rank_prefers_title_relevance():
    papers = [
        Paper("1", "KV cache compression for LLMs", [], "efficient cache quantization", datetime.now(timezone.utc), None, [], "", ""),
        Paper("2", "Image classification", [], "vision model", datetime.now(timezone.utc), None, [], "", ""),
    ]
    assert rank_candidates("KV-cache compression", papers)[0].arxiv_id == "1"
