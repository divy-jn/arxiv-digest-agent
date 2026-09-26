from __future__ import annotations

from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    query_type: Literal["paper", "topic"]
    arxiv_id: str
    candidates: list[object]
    selected_paper: object
    pdf_path: str
    parsed_sections: list[object]
    parse_valid: bool
    chunks: list[object]
    collection_name: str
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
