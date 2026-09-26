from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import requests
import time

from app.exceptions import ParseError, RetrievalError
from app.pdf.sections import detect_section


@dataclass(frozen=True)
class ExtractedBlock:
    text: str
    page: int
    section: str
    subsection: str | None = None


def download_pdf(url: str, destination: Path, timeout_seconds: int = 30) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    response: requests.Response | None = None
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.3)
    if response is None or (last_error is not None and not response.ok):
        raise RetrievalError(f"Could not download paper PDF after retry: {last_error}")
    if not response.content.startswith(b"%PDF"):
        raise RetrievalError("arXiv did not return a valid PDF.")
    destination.write_bytes(response.content)
    return destination


def extract_pdf(pdf_path: Path) -> list[ExtractedBlock]:
    try:
        document = fitz.open(pdf_path)
    except (fitz.FileDataError, OSError) as exc:
        raise ParseError(f"Cannot open PDF: {exc}") from exc
    blocks: list[ExtractedBlock] = []
    current_section = "Abstract"
    try:
        for page_index, page in enumerate(document):
            for raw_block in page.get_text("blocks", sort=True):
                text = " ".join(raw_block[4].split())
                if not text:
                    continue
                heading = detect_section(text)
                if heading:
                    current_section = heading
                    continue
                blocks.append(ExtractedBlock(text=text, page=page_index + 1, section=current_section))
    finally:
        document.close()
    validate_extraction(blocks)
    return blocks


def validate_extraction(blocks: list[ExtractedBlock]) -> None:
    chars = sum(len(block.text) for block in blocks)
    pages = max((block.page for block in blocks), default=0)
    if not blocks or chars < 800:
        raise ParseError("PDF extraction produced too little text; the paper may be scanned or malformed.")
    if pages >= 8 and chars / pages < 350:
        raise ParseError("PDF extraction quality is too low for reliable grounded QA.")
