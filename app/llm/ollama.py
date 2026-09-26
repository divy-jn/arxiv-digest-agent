from __future__ import annotations

from app.config import Settings
from app.exceptions import ConfigurationError


class OllamaLLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_model or not settings.llm_api_key:
            raise ConfigurationError("LLM_API_KEY and LLM_MODEL are required for briefing generation and QA. Copy .env.example to .env.")
        # Ollama Cloud exposes an OpenAI-compatible /v1 endpoint.
        from langchain_openai import ChatOpenAI
        self._chat = ChatOpenAI(model=settings.llm_model, base_url=settings.llm_base_url, api_key=settings.llm_api_key, temperature=0)

    def invoke(self, prompt: str) -> str:
        return str(self._chat.invoke(prompt).content)
