from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime | None
    updated: datetime | None
    categories: list[str]
    abs_url: str
    pdf_url: str
    html_url: str | None = None

    @property
    def paper_id(self) -> str:
        return self.arxiv_id.replace("/", "_")
