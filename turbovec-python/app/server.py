"""turbovec RAG server — FastAPI + HTMX with streaming answers.

Run:
    cd turbovec-python
    uv run python app/server.py

Or with auto-reload:
    cd turbovec-python/app
    uv run uvicorn server:app --reload
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from turbovec.langchain import TurboQuantVectorStore

_HERE = Path(__file__).parent
CORPUS_PATH = _HERE / "data" / "corpus.txt"
INDEX_PATH = _HERE / "data" / "saved_index"
K = 3


class _Embeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

print("Loading embedding model...")
_embeddings = _Embeddings()

def _chunk_with_meta(text: str, source: str) -> tuple[list[str], list[dict]]:
    chunks = _splitter.split_text(text)
    metas = [{"source": source, "chunk": i} for i, _ in enumerate(chunks)]
    return chunks, metas


if INDEX_PATH.exists():
    print(f"Loading saved index from {INDEX_PATH}...")
    _store = TurboQuantVectorStore.load(INDEX_PATH, _embeddings)
    print(f"{len(_store._docs)} documents loaded.")
else:
    print(f"Indexing corpus from {CORPUS_PATH}...")
    chunks, metas = _chunk_with_meta(CORPUS_PATH.read_text(), CORPUS_PATH.name)
    _store = TurboQuantVectorStore.from_texts(chunks, _embeddings, metadatas=metas)
    print(f"{len(chunks)} chunks indexed.")

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

app = FastAPI(title="turbovec RAG")
templates = Jinja2Templates(directory=str(_HERE / "templates"))


def _doc_list_html() -> str:
    docs = list(_store._docs.values())
    items = "".join(
        f'<li><em>{html.escape(meta.get("source", "?"))} #{meta.get("chunk", 0) + 1}</em> '
        f'{html.escape(text[:80])}{"…" if len(text) > 80 else ""}</li>'
        for text, meta in docs
    )
    return (
        f'<div id="doc-list">'
        f'<small>{len(docs)} chunks in index</small>'
        f'<ul class="doc-list">{items}</ul>'
        f'</div>'
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/query")
async def query(request: Request):
    body = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return HTMLResponse("", status_code=400)

    docs = _store.similarity_search(question, k=K)
    sources = [
        {
            "text": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "chunk": d.metadata.get("chunk", 0),
        }
        for d in docs
    ]
    context = "\n".join(s["text"] for s in sources)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer concisely using only the context above."
    )

    async def generate():
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        async for chunk in _llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield json.dumps({"type": "token", "data": chunk.content}) + "\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/documents", response_class=HTMLResponse)
async def list_documents():
    return HTMLResponse(_doc_list_html())


@app.post("/documents", response_class=HTMLResponse)
async def add_documents(text: str = Form(...)):
    chunks, metas = _chunk_with_meta(text, "manual")
    if chunks:
        _store.add_texts(chunks, metadatas=metas)
    return HTMLResponse(_doc_list_html())


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile):
    content = (await file.read()).decode(errors="replace")
    chunks, metas = _chunk_with_meta(content, file.filename or "upload")
    if chunks:
        _store.add_texts(chunks, metadatas=metas)
    return HTMLResponse(_doc_list_html())


@app.post("/save", response_class=HTMLResponse)
async def save_index():
    _store.dump(INDEX_PATH)
    return HTMLResponse('<span class="save-ok">Index saved.</span>')


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
