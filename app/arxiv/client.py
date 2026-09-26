from __future__ import annotations

import re
import time
from datetime import datetime
from xml.etree import ElementTree

import requests

from app.arxiv.models import Paper
from app.exceptions import InputError, RetrievalError

API_URL = "https://export.arxiv.org/api/query"
ID_PATTERN = re.compile(r"(?:https?://arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?$", re.IGNORECASE)
ATOM = "{http://www.w3.org/2005/Atom}"


def normalize_arxiv_id(value: str) -> str | None:
    """Return a canonical arXiv id for an id, abs URL, or pdf URL."""
    match = ID_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    return match.group(1)


def _text(element: ElementTree.Element, tag: str) -> str:
    return (element.findtext(f"{ATOM}{tag}") or "").strip().replace("\n", " ")


def _parse_date(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_entries(xml: str) -> list[Paper]:
    root = ElementTree.fromstring(xml)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _text(entry, "id")
        arxiv_id = normalize_arxiv_id(raw_id)
        if not arxiv_id:
            continue
        links = entry.findall(f"{ATOM}link")
        pdf_url = next((link.attrib["href"] for link in links if link.attrib.get("title") == "pdf"), f"https://arxiv.org/pdf/{arxiv_id}")
        papers.append(Paper(
            arxiv_id=arxiv_id, title=_text(entry, "title"),
            authors=[(author.findtext(f"{ATOM}name") or "").strip() for author in entry.findall(f"{ATOM}author")],
            summary=_text(entry, "summary"), published=_parse_date(_text(entry, "published")),
            updated=_parse_date(_text(entry, "updated")),
            categories=[category.attrib["term"] for category in entry.findall(f"{ATOM}category") if category.attrib.get("term")],
            abs_url=f"https://arxiv.org/abs/{arxiv_id}", pdf_url=pdf_url,
        ))
    return papers


class ArxivClient:
    def __init__(self, timeout_seconds: int = 20, session: requests.Session | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _request(self, params: dict[str, str | int]) -> list[Paper]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.session.get(API_URL, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return parse_entries(response.text)
            except (requests.RequestException, ElementTree.ParseError) as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(0.3)
        raise RetrievalError(f"arXiv API request failed after retry: {last_error}")

    def get_paper(self, identifier: str) -> Paper:
        arxiv_id = normalize_arxiv_id(identifier)
        if not arxiv_id:
            raise InputError("Enter an arXiv id (e.g. 2401.12345), arXiv URL, or research topic.")
        papers = self._request({"id_list": arxiv_id, "max_results": 1})
        if not papers:
            raise RetrievalError(f"No arXiv paper found for '{arxiv_id}'.")
        return papers[0]

    def search(self, topic: str, limit: int = 15) -> list[Paper]:
        papers = self._request({"search_query": f"all:{topic}", "start": 0, "max_results": limit, "sortBy": "submittedDate", "sortOrder": "descending"})
        unique = {paper.arxiv_id: paper for paper in papers}
        return list(unique.values())
