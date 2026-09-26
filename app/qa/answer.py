from __future__ import annotations


REFUSAL = "I couldn't find sufficient evidence in the paper to answer that."


def answer_from_evidence(llm: object, question: str, chunks: list[dict[str, object]]) -> str:
    evidence = "\n\n".join(f"[Section {chunk['metadata'].get('section')}, p. {chunk['metadata'].get('page_start')}]\n{chunk['text']}" for chunk in chunks)
    return llm.invoke(f"""Answer the question based ONLY on the provided excerpts. Do not use outside knowledge. If the excerpts contain the answer conceptually or explicitly, provide it. If the excerpts do not contain sufficient information to answer the question, reply exactly: {REFUSAL}

Question: {question}\n\nEvidence:\n{evidence}""")


def rewrite_question(llm: object, question: str) -> str:
    return llm.invoke(f"Rewrite this as a short paper-specific retrieval query. Return only the query: {question}")
