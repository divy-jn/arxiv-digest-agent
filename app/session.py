from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from app.config import Settings


def save_session(settings: Settings, state: dict) -> str:
    session_id = str(uuid.uuid4())
    paper = state["selected_paper"]
    briefing = state["briefing"]
    payload = {"session_id": session_id, "paper_id": paper.paper_id, "arxiv_id": paper.arxiv_id,
               "created_at": datetime.now(timezone.utc).isoformat(), "briefing": asdict(briefing),
               "conversation_history": state.get("conversation_history", [])}
    (settings.session_dir / f"{session_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return session_id


def update_conversation(settings: Settings, session_id: str, history: list[dict[str, str]]) -> None:
    """Persist the small conversational audit trail without re-indexing the paper."""
    path = settings.session_dir / f"{session_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Session '{session_id}' does not exist.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conversation_history"] = history
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
