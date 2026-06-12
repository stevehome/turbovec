"""End-to-end RAG test: sentence-transformers embeddings + turbovec + Claude.

Requires:
    uv pip install sentence-transformers langchain-anthropic "langchain-core>=0.3"
    export ANTHROPIC_API_KEY=...

Retrieval tests run without an API key. LLM tests are skipped when
ANTHROPIC_API_KEY is not set.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("langchain_core")

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from turbovec.langchain import TurboQuantVectorStore

needs_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

CORPUS = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python uses indentation to define code blocks instead of braces.",
    "Python is dynamically typed and supports multiple programming paradigms.",
    "The Python Package Index (PyPI) hosts over 400,000 packages.",
    "Python's GIL limits true multi-threading but multiprocessing works around it.",
    "Rust was created by Graydon Hoare and sponsored by Mozilla Research.",
    "Rust uses an ownership model to guarantee memory safety without a garbage collector.",
    "Cargo is Rust's built-in package manager and build system.",
    "Rust's borrow checker enforces memory safety at compile time.",
    "Rust is known for zero-cost abstractions and high performance.",
]


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


@pytest.fixture(scope="module")
def store():
    return TurboQuantVectorStore.from_texts(CORPUS, SentenceTransformerEmbeddings())


@pytest.fixture(scope="module")
def llm():
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=256)


# ---- Retrieval (no API key needed) ----------------------------------------

def test_retrieval_finds_python_creation_doc(store):
    results = store.similarity_search("When was Python first released?", k=2)
    contents = [r.page_content for r in results]
    assert any("1991" in c for c in contents), f"Expected 1991 in results, got: {contents}"


def test_retrieval_finds_rust_cargo_doc(store):
    results = store.similarity_search("What is Cargo in Rust?", k=2)
    contents = [r.page_content for r in results]
    assert any("Cargo" in c for c in contents), f"Expected Cargo doc in results, got: {contents}"


def test_retrieval_separates_python_from_rust(store):
    python_results = store.similarity_search("Python programming language", k=3)
    rust_results = store.similarity_search("Rust programming language", k=3)
    python_contents = " ".join(r.page_content for r in python_results)
    rust_contents = " ".join(r.page_content for r in rust_results)
    assert "Python" in python_contents
    assert "Rust" in rust_contents


# ---- RAG with LLM (requires ANTHROPIC_API_KEY) ----------------------------

@needs_api_key
def test_rag_answer_mentions_python_release_year(store, llm):
    from langchain_core.messages import HumanMessage
    question = "What year was Python first released?"
    docs = store.similarity_search(question, k=2)
    context = "\n".join(d.page_content for d in docs)
    response = llm.invoke([
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\nAnswer in one sentence.")
    ])
    assert "1991" in response.content, f"Expected 1991 in answer: {response.content}"


@needs_api_key
def test_rag_answer_explains_rust_memory_safety(store, llm):
    from langchain_core.messages import HumanMessage
    question = "How does Rust guarantee memory safety?"
    docs = store.similarity_search(question, k=2)
    context = "\n".join(d.page_content for d in docs)
    response = llm.invoke([
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}\nAnswer in one sentence.")
    ])
    answer = response.content.lower()
    assert any(kw in answer for kw in ("ownership", "borrow", "compile")), (
        f"Expected memory-safety terms in answer: {response.content}"
    )
