from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (Services, arxiv_search, check_evidence, corrective_rewrite, direct_paper, fetch_pdf,
    grounded_answer, index_chroma, make_briefing, make_chunks, parse_pdf, query_understanding, rank_and_select,
    retrieve_chunks, validate_parse, retry_parse, parse_failure, refuse)
from app.graph.state import AgentState


def _query_route(state: AgentState) -> str:
    return "direct_paper" if state["query_type"] == "paper" else "arxiv_search"


def _evidence_route(state: AgentState) -> str:
    if state["evidence_status"] == "strong":
        return "answer"
    return "rewrite" if state.get("retry_count", 0) < 1 else "refuse"


def _parse_route(state: AgentState) -> str:
    if state.get("parse_valid"):
        return "chunk"
    return "retry" if state.get("retry_count", 0) < 1 else "fail"


def build_ingestion_graph(services: Services):
    graph = StateGraph(AgentState)
    graph.add_node("query_understanding", query_understanding)
    graph.add_node("direct_paper", direct_paper(services))
    graph.add_node("arxiv_search", arxiv_search(services))
    graph.add_node("rank_candidates", rank_and_select)
    graph.add_node("fetch_pdf", fetch_pdf(services))
    graph.add_node("parse_pdf", parse_pdf)
    graph.add_node("validate_parse", validate_parse)
    graph.add_node("retry_parse", retry_parse)
    graph.add_node("parse_failure", parse_failure)
    graph.add_node("chunk_paper", make_chunks)
    graph.add_node("index_chroma", index_chroma(services))
    graph.add_node("generate_briefing", make_briefing(services))
    graph.add_edge(START, "query_understanding")
    graph.add_conditional_edges("query_understanding", _query_route, {"direct_paper": "direct_paper", "arxiv_search": "arxiv_search"})
    graph.add_edge("arxiv_search", "rank_candidates")
    graph.add_edge("direct_paper", "fetch_pdf")
    graph.add_edge("rank_candidates", "fetch_pdf")
    graph.add_edge("fetch_pdf", "parse_pdf")
    graph.add_edge("parse_pdf", "validate_parse")
    graph.add_conditional_edges("validate_parse", _parse_route, {"chunk": "chunk_paper", "retry": "retry_parse", "fail": "parse_failure"})
    graph.add_edge("retry_parse", "validate_parse")
    graph.add_edge("parse_failure", END)
    graph.add_edge("chunk_paper", "index_chroma")
    graph.add_edge("index_chroma", "generate_briefing")
    graph.add_edge("generate_briefing", END)
    return graph.compile()


def build_qa_graph(services: Services):
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_chunks", retrieve_chunks(services))
    graph.add_node("evidence_check", check_evidence)
    graph.add_node("rewrite_query", corrective_rewrite(services))
    graph.add_node("answer", grounded_answer(services))
    graph.add_node("refuse", refuse)
    graph.add_edge(START, "retrieve_chunks")
    graph.add_edge("retrieve_chunks", "evidence_check")
    graph.add_conditional_edges("evidence_check", _evidence_route, {"answer": "answer", "rewrite": "rewrite_query", "refuse": "refuse"})
    graph.add_edge("rewrite_query", "retrieve_chunks")
    graph.add_edge("answer", END)
    graph.add_edge("refuse", END)
    return graph.compile()
