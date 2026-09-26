from __future__ import annotations

from pathlib import Path

from app.arxiv.client import ArxivClient, normalize_arxiv_id
from app.arxiv.search import rank_candidates
from app.briefing.generator import generate_briefing
from app.config import Settings
from app.exceptions import InputError
from app.pdf.chunker import chunk_blocks
from app.pdf.extractor import download_pdf, extract_pdf
from app.qa.answer import REFUSAL, answer_from_evidence, rewrite_question
from app.qa.citations import format_citations
from app.qa.evidence import evidence_score, is_strong_evidence
from app.retrieval.chroma import ChromaStore


class Services:
    def __init__(self, settings: Settings, arxiv: ArxivClient, store: ChromaStore, llm: object) -> None:
        self.settings, self.arxiv, self.store, self.llm = settings, arxiv, store, llm


def query_understanding(state: dict, _: object = None) -> dict:
    candidate = normalize_arxiv_id(state["user_input"])
    return {"query_type": "paper" if candidate else "topic", "arxiv_id": candidate or ""}


def direct_paper(services: Services):
    return lambda state: {"selected_paper": services.arxiv.get_paper(state["arxiv_id"])}


def arxiv_search(services: Services):
    def run(state: dict) -> dict:
        topic = state["user_input"].strip()
        if len(topic) < 3:
            raise InputError("Please enter a more specific research topic.")
        candidates = services.arxiv.search(topic)
        if not candidates:
            raise InputError("No arXiv papers matched that topic. Try different terms.")
        return {"candidates": candidates}
    return run


def rank_and_select(state: dict) -> dict:
    ranked = rank_candidates(state["user_input"], state["candidates"])
    return {"candidates": ranked, "selected_paper": ranked[0]}


def fetch_pdf(services: Services):
    def run(state: dict) -> dict:
        paper = state["selected_paper"]
        path = services.settings.data_dir / "papers" / f"{paper.paper_id}.pdf"
        return {"pdf_path": str(download_pdf(paper.pdf_url, path))}
    return run


def parse_pdf(state: dict) -> dict:
    try:
        return {"parsed_sections": extract_pdf(Path(state["pdf_path"])), "parse_valid": True}
    except Exception as exc:
        return {"parse_valid": False, "errors": [*state.get("errors", []), str(exc)]}


def validate_parse(state: dict) -> dict:
    # extract_pdf validates extraction. This explicit graph node makes the valid/retry/fail branch auditable.
    return {"parse_valid": bool(state.get("parse_valid")), "retry_count": state.get("retry_count", 0)}


def retry_parse(state: dict) -> dict:
    """One bounded parse retry handles transient filesystem/PDF-reader failures."""
    try:
        return {"parsed_sections": extract_pdf(Path(state["pdf_path"])), "parse_valid": True, "retry_count": state.get("retry_count", 0) + 1}
    except Exception as exc:
        return {"parse_valid": False, "retry_count": state.get("retry_count", 0) + 1, "errors": [*state.get("errors", []), str(exc)]}


def parse_failure(state: dict) -> dict:
    detail = state.get("errors", ["unknown extraction error"])[-1]
    raise InputError(f"PDF parsing failed after retry: {detail}")


def make_chunks(state: dict) -> dict:
    paper = state["selected_paper"]
    return {"chunks": chunk_blocks(state["parsed_sections"], paper.paper_id, paper.arxiv_id)}


def index_chroma(services: Services):
    def run(state: dict) -> dict:
        services.store.index(state["chunks"])
        return {"collection_name": "arxiv_papers"}
    return run


def make_briefing(services: Services):
    def run(state: dict) -> dict:
        for _ in range(2):
            try:
                return {"briefing": generate_briefing(services.llm, state["selected_paper"], state["chunks"])}
            except ValueError:
                continue
        raise ValueError("Could not generate a complete structured briefing after retry.")
    return run


def retrieve_chunks(services: Services):
    def run(state: dict) -> dict:
        question = state.get("rewritten_question") or state["question"]
        paper = state["selected_paper"]
        chunks = services.store.query(question, paper.paper_id)
        return {"retrieved_chunks": chunks, "evidence_score": evidence_score(question, chunks)}
    return run


def check_evidence(state: dict) -> dict:
    question = state.get("rewritten_question") or state["question"]
    return {"evidence_status": "strong" if is_strong_evidence(question, state["retrieved_chunks"]) else "weak"}


def corrective_rewrite(services: Services):
    return lambda state: {"rewritten_question": rewrite_question(services.llm, state["question"]), "retry_count": state.get("retry_count", 0) + 1}


def grounded_answer(services: Services):
    def run(state: dict) -> dict:
        answer = answer_from_evidence(services.llm, state["question"], state["retrieved_chunks"])
        if answer.strip() == REFUSAL:
            return refuse(state)
        return {"answer": answer, "citations": format_citations(state["retrieved_chunks"])}
    return run


def refuse(_: dict) -> dict:
    return {"answer": REFUSAL, "citations": []}
