"""turbovec RAG demo — Gradio web UI.

Loads a plain-text corpus (one document per line), indexes it with
TurboQuantVectorStore, and answers questions using retrieved context and Claude.

Run:
    uv run python app/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer

from turbovec.langchain import TurboQuantVectorStore

CORPUS_PATH = Path(__file__).parent / "data" / "corpus.txt"
K = 3


class _Embeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def _load_corpus(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


print("Loading embedding model...")
_embeddings = _Embeddings()

print(f"Indexing corpus from {CORPUS_PATH}...")
_lines = _load_corpus(CORPUS_PATH)
_store = TurboQuantVectorStore.from_texts(_lines, _embeddings)
print(f"{len(_lines)} documents indexed.")

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=512)


def ask(question: str) -> tuple[str, str]:
    if not question.strip():
        return "", ""
    docs = _store.similarity_search(question, k=K)
    context = "\n".join(d.page_content for d in docs)
    response = _llm.invoke([
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely using only the context above.")
    ])
    sources = "\n".join(f"• {d.page_content}" for d in docs)
    return response.content, sources


with gr.Blocks(title="turbovec RAG demo") as demo:
    gr.Markdown(f"## turbovec RAG demo\n`{CORPUS_PATH.name}` — {len(_lines)} documents indexed")

    question = gr.Textbox(
        label="Question",
        placeholder="e.g. How does Rust handle memory safety?",
        lines=1,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    answer_out = gr.Textbox(label="Answer", lines=4, interactive=False)
    sources_out = gr.Textbox(label="Retrieved sources", lines=4, interactive=False)

    ask_btn.click(ask, inputs=question, outputs=[answer_out, sources_out])
    question.submit(ask, inputs=question, outputs=[answer_out, sources_out])


if __name__ == "__main__":
    demo.launch()
