from unittest.mock import MagicMock
import pytest
from app.graph.graph import _query_route, _evidence_route, _parse_route
from app.retrieval.chroma import ChromaStore
from app.briefing.generator import generate_briefing
from app.session import save_session, update_conversation
from app.config import Settings
from pathlib import Path
from dataclasses import dataclass
from app.arxiv.models import Paper
import datetime

def test_graph_query_routing():
    assert _query_route({"query_type": "paper"}) == "direct_paper"
    assert _query_route({"query_type": "topic"}) == "arxiv_search"

def test_parse_retry_failure_routing():
    assert _parse_route({"parse_valid": True}) == "chunk"
    assert _parse_route({"parse_valid": False, "retry_count": 0}) == "retry"
    assert _parse_route({"parse_valid": False, "retry_count": 1}) == "fail"

def test_corrective_qa_routing():
    assert _evidence_route({"evidence_status": "strong"}) == "answer"
    assert _evidence_route({"evidence_status": "weak", "retry_count": 0}) == "rewrite"
    assert _evidence_route({"evidence_status": "weak", "retry_count": 1}) == "refuse"

def test_chroma_active_paper_filtering():
    embedder = MagicMock()
    embedder.encode.return_value = [[0.1, 0.2]]
    store = ChromaStore(Path("/tmp/chroma"), embedder)
    store._collection = MagicMock()
    store._collection.query.return_value = {"ids": [["1"]], "metadatas": [[{"text": "A"}]], "documents": [["A"]], "distances": [[0.5]]}
    store.query("hello", "paper-123")
    store._collection.query.assert_called_once()
    args, kwargs = store._collection.query.call_args
    assert kwargs["where"] == {"paper_id": "paper-123"}

def _dummy_paper():
    return Paper(
        arxiv_id="123",
        title="title",
        authors=["author"],
        summary="sum",
        published=datetime.datetime.now(),
        updated=datetime.datetime.now(),
        categories=["cs.AI"],
        abs_url="url",
        pdf_url="url"
    )

def test_briefing_required_field_validation():
    llm = MagicMock()
    paper = _dummy_paper()
    # First returns invalid (missing key_results), second returns valid
    llm.invoke.side_effect = [
        '```json\n{"why_it_matters": "A", "problem_statement": "B", "method": "C", "limitations": [], "follow_up_questions": []}\n```',
        '```json\n{"why_it_matters": "A", "problem_statement": "B", "method": "C", "key_results": [], "limitations": [], "follow_up_questions": []}\n```'
    ]
    with pytest.raises(ValueError, match="Ollama briefing omitted required fields."):
        generate_briefing(llm, paper, [])
    # Valid call should pass
    b = generate_briefing(llm, paper, [])
    assert b.why_it_matters == "A"

def test_session_persistence(tmp_path):
    settings = MagicMock()
    settings.session_dir = tmp_path
    from app.briefing.generator import Briefing
    b = Briefing("title", [], "123", "date", "link", "A", "B", "C", [], [], [])
    session_id = save_session(settings, {"briefing": b, "selected_paper": _dummy_paper()})
    assert (tmp_path / f"{session_id}.json").exists()
    update_conversation(settings, session_id, [{"role": "user", "content": "hello"}])
    import json
    data = json.loads((tmp_path / f"{session_id}.json").read_text())
    assert len(data["conversation_history"]) == 1
    assert data["conversation_history"][0]["content"] == "hello"
