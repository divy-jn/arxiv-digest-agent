from __future__ import annotations

import re

SECTION_RE = re.compile(r"^(?:(\d+(?:\.\d+)*)\s+)?([A-Z][A-Za-z][A-Za-z \-&]{2,80})$")


def detect_section(line: str) -> str | None:
    cleaned = " ".join(line.split())
    lowered = cleaned.lower()
    if lowered in {"abstract", "references", "introduction", "conclusion", "conclusions", "related work", "acknowledgments", "acknowledgements"}:
        return cleaned.title()
    match = SECTION_RE.match(cleaned)
    if match and len(cleaned.split()) <= 10:
        return match.group(2).strip().title()
    return None
