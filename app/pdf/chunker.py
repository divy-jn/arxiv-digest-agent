from __future__ import annotations

from dataclasses import dataclass

from app.pdf.extractor import ExtractedBlock


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    paper_id: str
    arxiv_id: str
    section: str
    subsection: str | None
    page_start: int
    page_end: int


def chunk_blocks(blocks: list[ExtractedBlock], paper_id: str, arxiv_id: str, target_tokens: int = 800, overlap_tokens: int = 100) -> list[Chunk]:
    """Keep paragraphs/sections intact where possible; overlap only at boundaries."""
    max_words = target_tokens * 4
    overlap_words = overlap_tokens * 4
    chunks: list[Chunk] = []
    bucket: list[ExtractedBlock] = []
    word_count = 0

    def flush() -> None:
        nonlocal bucket, word_count
        if not bucket:
            return
        text = "\n\n".join(block.text for block in bucket)
        chunks.append(Chunk(id=f"{paper_id}-{len(chunks):04d}", text=text, paper_id=paper_id, arxiv_id=arxiv_id,
                            section=bucket[0].section, subsection=bucket[0].subsection,
                            page_start=bucket[0].page, page_end=bucket[-1].page))
        tail: list[ExtractedBlock] = []
        count = 0
        for block in reversed(bucket):
            tail.insert(0, block)
            count += len(block.text.split())
            if count >= overlap_words:
                break
        bucket = tail
        word_count = sum(len(block.text.split()) for block in bucket)

    for block in blocks:
        words = len(block.text.split())
        if bucket and (block.section != bucket[0].section or word_count + words > max_words):
            flush()
            if bucket and block.section != bucket[0].section:
                bucket, word_count = [], 0
        bucket.append(block)
        word_count += words
    if bucket:
        text = "\n\n".join(block.text for block in bucket)
        chunks.append(Chunk(id=f"{paper_id}-{len(chunks):04d}", text=text, paper_id=paper_id, arxiv_id=arxiv_id,
                            section=bucket[0].section, subsection=bucket[0].subsection,
                            page_start=bucket[0].page, page_end=bucket[-1].page))
    return chunks
