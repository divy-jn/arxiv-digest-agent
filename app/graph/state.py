from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    query_type: Literal["paper", "topic"]
    input_kind: Literal["id", "abs_url", "pdf_url", "html_url"]
    paper_version: str | None
    arxiv_id: str
    candidates: list[object]
    selected_paper: object
    html_url: str | None
    source_type: Literal["pdf", "html"]
    pdf_path: str
    parsed_sections: list[object]
    parse_valid: bool
    chunks: list[object]
    collection_name: str
    ingestion_cached: bool
    briefing: object
    question: str
    rewritten_question: str
    retrieved_chunks: list[dict[str, object]]
    evidence_status: Literal["strong", "weak"]
    evidence_score: float
    answer: str
    citations: list[str]
    conversation_history: list[dict[str, str]]
    retry_count: int
    errors: list[str]
