from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    chroma_dir: Path
    session_dir: Path
    embedding_model: str
    llm_model: str | None
    llm_api_key: str | None
    llm_base_url: str
    arxiv_timeout_seconds: int

    @classmethod
    def load(cls, root: Path | None = None) -> "Settings":
        load_dotenv()
        project_root = root or Path(__file__).resolve().parents[1]
        data_dir = project_root / "data"
        return cls(
            root=project_root,
            data_dir=data_dir,
            chroma_dir=data_dir / "chroma",
            session_dir=data_dir / "sessions",
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            llm_model=os.getenv("LLM_MODEL") or None,
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://ollama.com/v1"),
            arxiv_timeout_seconds=int(os.getenv("ARXIV_TIMEOUT_SECONDS", "20")),
        )

    def ensure_data_dirs(self) -> None:
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
