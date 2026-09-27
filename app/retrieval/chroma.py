from __future__ import annotations

from pathlib import Path

from app.pdf.chunker import Chunk


class ChromaStore:
    def __init__(self, directory: Path, embedding_function: object) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=str(directory))
        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(name="arxiv_papers", metadata={"hnsw:space": "cosine"})

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(ids=[chunk.id for chunk in chunks], documents=[chunk.text for chunk in chunks],
            metadatas=[{"paper_id": chunk.paper_id, "arxiv_id": chunk.arxiv_id, "section": chunk.section,
                        "subsection": chunk.subsection or "", "page_start": chunk.page_start, "page_end": chunk.page_end} for chunk in chunks],
            embeddings=self._embedding_function.encode([chunk.text for chunk in chunks]))

    def has_paper(self, paper_id: str) -> bool:
        """Check whether any chunks for this paper_id already exist in the collection.

        Uses a count-only metadata query — no embedding computation required.
        Returns True if at least one chunk is indexed for the given paper_id.
        """
        result = self._collection.get(where={"paper_id": paper_id}, limit=1, include=[])
        return bool(result and result["ids"])

    def get_chunks(self, paper_id: str) -> list[Chunk]:
        result = self._collection.get(where={"paper_id": paper_id}, include=["documents", "metadatas"])
        if not result or not result["ids"]:
            return []
        
        chunks = []
        for i in range(len(result["ids"])):
            meta = result["metadatas"][i]
            chunks.append(Chunk(
                id=result["ids"][i],
                text=result["documents"][i],
                paper_id=meta["paper_id"],
                arxiv_id=meta["arxiv_id"],
                section=meta["section"],
                subsection=meta.get("subsection"),
                page_start=meta["page_start"],
                page_end=meta["page_end"]
            ))
        return chunks

    def query(self, question: str, paper_id: str, limit: int = 8) -> list[dict[str, object]]:
        result = self._collection.query(query_embeddings=self._embedding_function.encode([question]), n_results=limit,
                                        where={"paper_id": paper_id}, include=["documents", "metadatas", "distances"])
        return [{"id": result["ids"][0][i], "text": result["documents"][0][i], "metadata": result["metadatas"][0][i], "distance": result["distances"][0][i]}
                for i in range(len(result["ids"][0]))]
