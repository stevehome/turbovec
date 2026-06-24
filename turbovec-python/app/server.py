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
import json
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import enrich_chunks, extract_text
from store import (
    CORPUS_PATH, INDEX_PATH, SETTINGS_PATH, SOURCES_DIR,
    K, chunk_with_meta, delete_source_file, save_settings,
    save_source, save_sources, state,
)
from turbovec.langchain import TurboQuantVectorStore
from ui import doc_list_html

_HERE = Path(__file__).parent
app = FastAPI(title="turbovec RAG")
templates = Jinja2Templates(directory=str(_HERE / "templates"))


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
    filter_sources = [s for s in body.get("filter_sources", []) if s]

    if len(filter_sources) == 1:
        src_filter = {"source": filter_sources[0]}
    elif filter_sources:
        src_filter = lambda doc: doc.metadata.get("source") in filter_sources
    else:
        src_filter = None

    # Fetch more candidates than needed, then re-rank with a cross-encoder.
    fetch_k = min(len(state.store._docs), max(k * 4, 20))
    candidates = state.store.similarity_search_with_score(question, k=fetch_k, filter=src_filter)

    pairs = [(question, d.page_content) for d, _ in candidates]
    scores = await asyncio.to_thread(state.reranker.predict, pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:k]

    sources = [
        {"text": d.page_content, "source": d.metadata.get("source", "unknown"),
         "chunk": d.metadata.get("chunk", 0), "score": round(float(score), 3)}
        for score, (d, _) in ranked
    ]
    context = "\n".join(s["text"] for s in sources)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely using only the context above."

    async def generate():
        yield json.dumps({"type": "sources", "data": sources}) + "\n"
        async for chunk in state.llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield json.dumps({"type": "token", "data": chunk.content}) + "\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/documents", response_class=HTMLResponse)
async def list_documents():
    return HTMLResponse(doc_list_html())


@app.get("/sources/{source_name}/context")
async def source_context(source_name: str, chunk_index: int = Query(0), window: int = Query(500)):
    full_text = state.sources.get(source_name, "")
    chunk_text = next(
        (text for _, (text, meta) in state.store._docs.items()
         if meta.get("source") == source_name and meta.get("chunk") == chunk_index),
        None,
    )
    if chunk_text is None or not full_text:
        return JSONResponse({"error": "not found"}, status_code=404)
    needle = chunk_text
    pos = full_text.find(needle[:120])
    if pos == -1:
        original = chunk_text.split("\n\n", 1)[-1]
        pos = full_text.find(original[:120])
        needle = original
    if pos == -1:
        return JSONResponse({"error": "chunk not found in source"}, status_code=404)
    start = max(0, pos - window)
    end = min(len(full_text), pos + len(needle) + window)
    return JSONResponse({
        "before": full_text[start:pos], "chunk": needle,
        "after": full_text[pos + len(needle):end],
        "truncated_start": start > 0, "truncated_end": end < len(full_text),
    })


@app.delete("/documents/{doc_id}", response_class=HTMLResponse)
async def delete_document(doc_id: str):
    if doc_id in state.store._docs:
        state.store.delete([doc_id])
        state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.delete("/sources/{source_name}", response_class=HTMLResponse)
async def delete_source(source_name: str):
    ids = [sid for sid, (_, meta) in state.store._docs.items() if meta.get("source") == source_name]
    if ids:
        state.store.delete(ids)
        state.sources.pop(source_name, None)
        delete_source_file(source_name)
        state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/documents", response_class=HTMLResponse)
async def add_documents(text: str = Form(...)):
    chunks, metas = chunk_with_meta(text, "manual")
    if chunks:
        if state.contextual:
            chunks = await enrich_chunks(chunks, text)
        state.store.add_texts(chunks, metadatas=metas)
        state.sources["manual"] = (state.sources.get("manual", "") + "\n\n" + text).strip()
        save_source("manual", state.sources["manual"])
        state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile):
    data = await file.read()
    filename = file.filename or "upload"
    content = await asyncio.to_thread(extract_text, filename, data)
    chunks, metas = chunk_with_meta(content, filename)
    if chunks:
        if state.contextual:
            chunks = await enrich_chunks(chunks, content)
        state.store.add_texts(chunks, metadatas=metas)
        state.sources[filename] = content
        save_source(filename, content)
        state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/reindex", response_class=HTMLResponse)
async def reindex():
    old_ids = [sid for sid, (_, meta) in state.store._docs.items() if meta.get("source") == CORPUS_PATH.name]
    if old_ids:
        state.store.delete(old_ids)
    corpus_text = CORPUS_PATH.read_text()
    state.sources[CORPUS_PATH.name] = corpus_text
    chunks, metas = chunk_with_meta(corpus_text, CORPUS_PATH.name)
    if chunks:
        if state.contextual:
            chunks = await enrich_chunks(chunks, corpus_text)
        state.store.add_texts(chunks, metadatas=metas)
    save_source(CORPUS_PATH.name, corpus_text)
    state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/rechunk", response_class=HTMLResponse)
async def rechunk(chunk_size: int = Form(500), chunk_overlap: int = Form(50), contextual: str = Form("")):
    chunk_size = max(50, min(2000, chunk_size))
    chunk_overlap = max(0, min(chunk_size - 1, chunk_overlap))
    state.chunk_size    = chunk_size
    state.chunk_overlap = chunk_overlap
    state.contextual    = bool(contextual)
    state.splitter      = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    save_settings(chunk_size, chunk_overlap, state.contextual)

    for source, text in list(state.sources.items()):
        old_ids = [sid for sid, (_, meta) in state.store._docs.items() if meta.get("source") == source]
        if old_ids:
            state.store.delete(old_ids)
        chunks, metas = chunk_with_meta(text, source)
        if chunks:
            if state.contextual:
                chunks = await enrich_chunks(chunks, text)
            state.store.add_texts(chunks, metadatas=metas)

    state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/rebuild", response_class=HTMLResponse)
async def rebuild(bit_width: int = Form(4)):
    bit_width = max(2, min(4, bit_width))
    texts = [text for text, _meta in state.store._docs.values()]
    metas = [meta for _text, meta in state.store._docs.values()]
    state.store = TurboQuantVectorStore.from_texts(texts, state.embeddings, metadatas=metas, bit_width=bit_width)
    state.store.dump(INDEX_PATH)
    return HTMLResponse(doc_list_html())


@app.post("/clear", response_class=HTMLResponse)
async def clear_index():
    state.store   = TurboQuantVectorStore(state.embeddings)
    state.sources = {}
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)
    if SOURCES_DIR.exists():
        shutil.rmtree(SOURCES_DIR)
        SOURCES_DIR.mkdir()
    SETTINGS_PATH.unlink(missing_ok=True)
    return HTMLResponse(doc_list_html())


@app.post("/save", response_class=HTMLResponse)
async def save_index():
    state.store.dump(INDEX_PATH)
    return HTMLResponse('<span class="save-ok">Index saved.</span>')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
