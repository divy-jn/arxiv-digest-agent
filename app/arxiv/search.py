from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from app.arxiv.models import Paper


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def rank_candidates(topic: str, papers: list[Paper]) -> list[Paper]:
    """Rank with deterministic lexical relevance plus a small recency signal."""
    query = _tokens(topic)
    now = datetime.now(timezone.utc)

    def score(paper: Paper) -> float:
        title = _tokens(paper.title)
        abstract = _tokens(paper.summary)
        title_overlap = len(query & title) / max(1, len(query))
        abstract_overlap = len(query & abstract) / max(1, len(query))
        recency = 0.0
        if paper.published:
            published = paper.published if paper.published.tzinfo else paper.published.replace(tzinfo=timezone.utc)
            age_days = max(0, (now - published).days)
            recency = math.exp(-age_days / 730) * 0.08
        return title_overlap * 0.62 + abstract_overlap * 0.30 + recency

    return sorted(papers, key=lambda paper: (score(paper), paper.published or datetime.min), reverse=True)
