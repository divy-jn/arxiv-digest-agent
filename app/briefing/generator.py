from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from app.arxiv.models import Paper
from app.pdf.chunker import Chunk


@dataclass(frozen=True)
class Briefing:
    title: str
    authors: list[str]
    arxiv_id: str
    publish_date: str
    paper_link: str
    why_it_matters: str
    problem_statement: str
    method: str
    key_results: list[str]
    limitations: list[str]
    follow_up_questions: list[str]

    def to_markdown(self) -> str:
        results = "\n".join(f"- {item}" for item in self.key_results)
        limitations = "\n".join(f"- {item}" for item in self.limitations)
        questions = "\n".join(f"- {item}" for item in self.follow_up_questions)
        return f"# {self.title}\n\n**Authors:** {', '.join(self.authors)}  \n**arXiv:** {self.arxiv_id}  \n**Published:** {self.publish_date}  \n**Link:** {self.paper_link}\n\n## Why this paper matters\n{self.why_it_matters}\n\n## Problem statement\n{self.problem_statement}\n\n## Method / approach\n{self.method}\n\n## Key results / claims\n{results}\n\n## Limitations\n{limitations}\n\n## Suggested follow-up questions\n{questions}"


REQUIRED = {"why_it_matters", "problem_statement", "method", "key_results", "limitations", "follow_up_questions"}


def generate_briefing(llm: object, paper: Paper, chunks: list[Chunk]) -> Briefing:
    """Hierarchical summary: summarize representative section evidence, then synthesize."""
    by_section: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section, []).append(chunk)
    section_evidence = "\n\n".join(f"[{section}]\n{section_chunks[0].text[:3500]}" for section, section_chunks in by_section.items())
    prompt = f"""You are summarizing ONLY the supplied arXiv paper evidence. Return valid JSON with keys: why_it_matters, problem_statement, method, key_results (array), limitations (array), follow_up_questions (array). Do not add facts, measurements, or limitations absent from evidence. If limitations are not stated, say that explicitly as one item.

Paper metadata: {paper.title}\nEvidence:\n{section_evidence[:18000]}"""
    raw = llm.invoke(prompt)
    try:
        payload = json.loads(raw.removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Ollama returned malformed briefing JSON.") from exc
    if not REQUIRED.issubset(payload) or not isinstance(payload["limitations"], list):
        raise ValueError("Ollama briefing omitted required fields.")
    return Briefing(title=paper.title, authors=paper.authors, arxiv_id=paper.arxiv_id,
        publish_date=paper.published.date().isoformat() if paper.published else "Unknown", paper_link=paper.abs_url,
        why_it_matters=payload["why_it_matters"], problem_statement=payload["problem_statement"], method=payload["method"],
        key_results=payload["key_results"], limitations=payload["limitations"], follow_up_questions=payload["follow_up_questions"])
