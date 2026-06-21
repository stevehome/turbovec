"""turbovec RAG server — FastAPI + HTMX with streaming answers.

Run:
    cd turbovec-python
    uv run python app/server.py

Or with auto-reload:
    cd turbovec-python/app
    uv run uvicorn server:app --reload
"""
from __future__ import annotations

import asyncio
import html
import io
import json
from pathlib import Path

import psutil

from dotenv import load_dotenv

load_dotenv()

import pypdf
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
SETTINGS_PATH = _HERE / "data" / "settings.json"
SOURCES_PATH = _HERE / "data" / "sources.json"
K = 3


def _load_settings() -> tuple[int, int, bool]:
    if SETTINGS_PATH.exists():
        s = json.loads(SETTINGS_PATH.read_text())
        return int(s.get("chunk_size", 500)), int(s.get("chunk_overlap", 50)), bool(s.get("contextual", False))
    return 500, 50, False


def _save_settings(chunk_size: int, chunk_overlap: int, contextual: bool) -> None:
    SETTINGS_PATH.write_text(json.dumps({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "contextual": contextual}))


def _load_sources() -> dict[str, str]:
    if SOURCES_PATH.exists():
        return json.loads(SOURCES_PATH.read_text())
    return {}


def _save_sources() -> None:
    SOURCES_PATH.write_text(json.dumps(_sources))


class _Embeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


_chunk_size, _chunk_overlap, _contextual = _load_settings()
_splitter = RecursiveCharacterTextSplitter(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
_sources: dict[str, str] = _load_sources()

print("Loading embedding model...")
_embeddings = _Embeddings()


def _chunk_with_meta(text: str, source: str) -> tuple[list[str], list[dict]]:
    chunks = _splitter.split_text(text)
    metas = [{"source": source, "chunk": i} for i, _ in enumerate(chunks)]
    return chunks, metas


async def _enrich_chunks(chunks: list[str], source_text: str) -> list[str]:
    """Prepend a one-sentence context to each chunk (Anthropic Contextual Retrieval).

    Calls are parallelised — N chunks = N concurrent haiku requests.
    """
    async def _one(chunk: str) -> str:
        prompt = (
            f"<document>\n{source_text}\n</document>\n\n"
            f"Here is the chunk we want to situate within the whole document:\n"
            f"<chunk>\n{chunk}\n</chunk>\n\n"
            "Give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval. Answer only with the succinct context and nothing else."
        )
        resp = await _llm.ainvoke([HumanMessage(content=prompt)])
        return f"{resp.content.strip()}\n\n{chunk}"

    return list(await asyncio.gather(*[_one(c) for c in chunks]))


if INDEX_PATH.exists():
    print(f"Loading saved index from {INDEX_PATH}...")
    _store = TurboQuantVectorStore.load(INDEX_PATH, _embeddings)
    print(f"{len(_store._docs)} documents loaded.")
    # Seed corpus.txt into _sources when loading a pre-source-tracking index.
    if CORPUS_PATH.name not in _sources and CORPUS_PATH.exists():
        _sources[CORPUS_PATH.name] = CORPUS_PATH.read_text()
        _save_sources()
else:
    print(f"Indexing corpus from {CORPUS_PATH}...")
    corpus_text = CORPUS_PATH.read_text()
    _sources[CORPUS_PATH.name] = corpus_text
    chunks, metas = _chunk_with_meta(corpus_text, CORPUS_PATH.name)
    _store = TurboQuantVectorStore.from_texts(chunks, _embeddings, metadatas=metas)
    _save_sources()
    print(f"{len(chunks)} chunks indexed.")

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

app = FastAPI(title="turbovec RAG")
templates = Jinja2Templates(directory=str(_HERE / "templates"))


def _memory_stats() -> str:
    n = len(_store._docs)
    dim = _store._index.dim
    bit_width = _store._index.bit_width
    if n == 0 or dim is None:
        vec_line = f"Vectors: empty ({bit_width}-bit)"
    else:
        q_bytes = n * dim * bit_width / 8
        fp32_bytes = n * dim * 4
        ratio = fp32_bytes / q_bytes

        def _fmt(b: float) -> str:
            return f"{b / (1024 * 1024):.1f} MB" if b >= 1024 * 1024 else f"{b / 1024:.1f} KB"

        vec_line = f"Vectors: {bit_width}-bit · {_fmt(q_bytes)} · {_fmt(fp32_bytes)} FP32 ({ratio:.0f}x)"
    proc_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    return f'{vec_line} · Process: {proc_mb:.0f} MB'


def _doc_list_html() -> str:
    # Group chunks by source, preserving insertion order.
    groups: dict[str, list[tuple[str, str, dict]]] = {}
    for sid, (text, meta) in _store._docs.items():
        source = meta.get("source", "?")
        groups.setdefault(source, []).append((sid, text, meta))

    sections = ""
    for source, chunks in groups.items():
        src_escaped = html.escape(source)
        src_url = html.escape(source, quote=True)
        rows = "".join(
            f'<li style="display:flex;align-items:baseline;gap:0.5rem">'
            f'<em>#{meta.get("chunk", 0) + 1}</em> '
            f'<span style="flex:1">{html.escape(text[:80])}{"…" if len(text) > 80 else ""}</span>'
            f'<button style="padding:0 0.3rem;font-size:0.72rem" class="secondary outline"'
            f' hx-delete="/documents/{sid}"'
            f' hx-target="#doc-list" hx-swap="outerHTML"'
            f' hx-confirm="Delete this chunk?">×</button>'
            f'</li>'
            for sid, text, meta in chunks
        )
        sections += (
            f'<details style="margin-bottom:0.5rem">'
            f'<summary style="display:flex;align-items:center;gap:0.5rem;cursor:pointer">'
            f'<strong>{src_escaped}</strong>'
            f'<small style="color:var(--pico-muted-color)">{len(chunks)} chunk{"s" if len(chunks) != 1 else ""}</small>'
            f'<button style="padding:0 0.4rem;font-size:0.72rem;margin-left:auto" class="secondary outline"'
            f' hx-delete="/sources/{src_url}"'
            f' hx-target="#doc-list" hx-swap="outerHTML"'
            f' hx-confirm="Delete all chunks from {src_escaped}?">delete source</button>'
            f'</summary>'
            f'<ul class="doc-list">{rows}</ul>'
            f'</details>'
        )

    total = len(_store._docs)
    return (
        f'<div id="doc-list">'
        f'<small>{total} chunk{"s" if total != 1 else ""} · {len(groups)} source{"s" if len(groups) != 1 else ""}</small>'
        f'{sections}'
        f'<p style="font-size:0.75rem;color:var(--pico-muted-color);margin:0.4rem 0 0">'
        f'{_memory_stats()} · chunk_size={_chunk_size} overlap={_chunk_overlap}'
        f'{" · contextual=on" if _contextual else ""}</p>'
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
    k = max(1, min(10, int(body.get("k", K))))

    docs = _store.similarity_search_with_score(question, k=k)
    sources = [
        {
            "text": d.page_content,
            "source": d.metadata.get("source", "unknown"),
            "chunk": d.metadata.get("chunk", 0),
            "score": round(score, 3),
        }
        for d, score in docs
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


@app.delete("/documents/{doc_id}", response_class=HTMLResponse)
async def delete_document(doc_id: str):
    if doc_id in _store._docs:
        _store.delete([doc_id])
        _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.delete("/sources/{source_name}", response_class=HTMLResponse)
async def delete_source(source_name: str):
    ids = [sid for sid, (_, meta) in _store._docs.items() if meta.get("source") == source_name]
    if ids:
        _store.delete(ids)
        _sources.pop(source_name, None)
        _save_sources()
        _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.post("/documents", response_class=HTMLResponse)
async def add_documents(text: str = Form(...)):
    chunks, metas = _chunk_with_meta(text, "manual")
    if chunks:
        if _contextual:
            chunks = await _enrich_chunks(chunks, text)
        _store.add_texts(chunks, metadatas=metas)
        _sources["manual"] = (_sources.get("manual", "") + "\n\n" + text).strip()
        _save_sources()
        _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode(errors="replace")


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile):
    data = await file.read()
    filename = file.filename or "upload"
    content = _extract_text(filename, data)
    chunks, metas = _chunk_with_meta(content, filename)
    if chunks:
        if _contextual:
            chunks = await _enrich_chunks(chunks, content)
        _store.add_texts(chunks, metadatas=metas)
        _sources[filename] = content
        _save_sources()
        _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.post("/reindex", response_class=HTMLResponse)
async def reindex():
    old_ids = [sid for sid, (_, meta) in _store._docs.items() if meta.get("source") == CORPUS_PATH.name]
    if old_ids:
        _store.delete(old_ids)
    corpus_text = CORPUS_PATH.read_text()
    _sources[CORPUS_PATH.name] = corpus_text
    chunks, metas = _chunk_with_meta(corpus_text, CORPUS_PATH.name)
    if chunks:
        if _contextual:
            chunks = await _enrich_chunks(chunks, corpus_text)
        _store.add_texts(chunks, metadatas=metas)
    _save_sources()
    _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.post("/rechunk", response_class=HTMLResponse)
async def rechunk(chunk_size: int = Form(500), chunk_overlap: int = Form(50), contextual: str = Form("")):
    global _splitter, _chunk_size, _chunk_overlap, _contextual
    chunk_size = max(50, min(2000, chunk_size))
    chunk_overlap = max(0, min(chunk_size - 1, chunk_overlap))
    _chunk_size, _chunk_overlap = chunk_size, chunk_overlap
    _contextual = bool(contextual)
    _splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    _save_settings(chunk_size, chunk_overlap, _contextual)

    # Re-chunk every source whose text we have stored.
    for source, text in list(_sources.items()):
        old_ids = [sid for sid, (_, meta) in _store._docs.items() if meta.get("source") == source]
        if old_ids:
            _store.delete(old_ids)
        chunks, metas = _chunk_with_meta(text, source)
        if chunks:
            if _contextual:
                chunks = await _enrich_chunks(chunks, text)
            _store.add_texts(chunks, metadatas=metas)

    _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.post("/rebuild", response_class=HTMLResponse)
async def rebuild(bit_width: int = Form(4)):
    global _store
    bit_width = max(2, min(4, bit_width))
    texts = [text for text, _meta in _store._docs.values()]
    metas = [meta for _text, meta in _store._docs.values()]
    _store = TurboQuantVectorStore.from_texts(texts, _embeddings, metadatas=metas, bit_width=bit_width)
    _store.dump(INDEX_PATH)
    return HTMLResponse(_doc_list_html())


@app.post("/save", response_class=HTMLResponse)
async def save_index():
    _store.dump(INDEX_PATH)
    return HTMLResponse('<span class="save-ok">Index saved.</span>')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
