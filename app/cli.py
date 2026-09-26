from __future__ import annotations

from app.arxiv.client import ArxivClient
from app.config import Settings
from app.exceptions import DigestError
from app.graph.graph import build_ingestion_graph, build_qa_graph
from app.graph.nodes import Services
from app.llm.ollama import OllamaLLM
from app.retrieval.chroma import ChromaStore
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.session import save_session, update_conversation


def main() -> None:
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print("\n================================================\n       ARXIV PAPER DIGEST AGENT\n================================================\n")
    user_input = input("Enter a topic or arXiv ID/URL: \n> ").strip()
    if not user_input:
        print("Please provide a topic or arXiv paper id.")
        return
    try:
        settings = Settings.load(); settings.ensure_data_dirs()
        services = Services(settings, ArxivClient(settings.arxiv_timeout_seconds),
            ChromaStore(settings.chroma_dir, SentenceTransformerEmbedder(settings.embedding_model)), OllamaLLM(settings))
        print("[+] Query understood\n[+] Searching arXiv / resolving paper\n[+] Downloading and parsing PDF\n[+] Building knowledge base\n[+] Generating briefing")
        state = build_ingestion_graph(services).invoke({"user_input": user_input, "retry_count": 0, "errors": []})
        print("\n------------------------------------------------\nEXECUTIVE BRIEFING\n------------------------------------------------\n")
        print(state["briefing"].to_markdown())
        session_id = save_session(settings, state)
        print(f"\nSession saved: {session_id}")
        qa_graph = build_qa_graph(services)
        history: list[dict[str, str]] = []
        print("\n------------------------------------------------\nQA MODE (type 'exit' to finish)\n------------------------------------------------")
        while question := input("\nAsk a question: \n> ").strip():
            if question.lower() in {"exit", "quit"}:
                break
            result = qa_graph.invoke({"selected_paper": state["selected_paper"], "question": question, "retry_count": 0})
            print(f"\nAnswer:\n{result['answer']}")
            if result.get("citations"):
                print("\nEvidence:\n" + "\n".join(f"- {citation}" for citation in result["citations"]))
            history.extend([{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}])
            update_conversation(settings, session_id, history)
    except DigestError as exc:
        print(f"\nUnable to continue: {exc}")
    except Exception as exc:
        print(f"\nUnexpected failure: {exc}")


if __name__ == "__main__":
    main()
