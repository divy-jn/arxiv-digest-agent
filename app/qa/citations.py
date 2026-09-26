from __future__ import annotations


def format_citations(chunks: list[dict[str, object]]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    citations: list[str] = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert isinstance(metadata, dict)
        section = str(metadata.get("section") or "Paper")
        page = int(metadata.get("page_start") or 1)
        key = (section, page)
        if key not in seen:
            citations.append(f"{section}, p. {page}")
            seen.add(key)
    return citations
