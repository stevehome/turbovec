# Plan: turbovec RAG Web App

## What's built

A local RAG web app: FastAPI + HTMX backend, streaming Claude answers, turbovec vector index.

**Stack**
- FastAPI + HTMX — server-rendered UI with streaming
- `sentence-transformers` (`all-MiniLM-L6-v2`) — local embeddings, no API cost
- `TurboQuantVectorStore` (langchain adapter) — quantized vector store
- `langchain-anthropic` (`claude-haiku-4-5`) — LLM for answers
- `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50) — chunking

**Run**
```bash
cd turbovec-python
uv run python app/server.py        # http://127.0.0.1:8000
# or with auto-reload:
cd turbovec-python/app && uv run uvicorn server:app --reload
```

## Features shipped

| Feature | Detail |
|---------|--------|
| Streaming answers | SSE-style newline-delimited JSON; sources sent first |
| Smart chunking | `RecursiveCharacterTextSplitter` on all ingestion paths |
| Chunk provenance | `{"source": filename, "chunk": i}` metadata on every chunk |
| Relevance scores | `similarity_search_with_score`; shown per source in UI |
| Low-confidence warning | Red banner when top score < 0.4 |
| K slider | 1–10 range input; sent as `k` in query POST body |
| Add text | Textarea → chunks → index; auto-saved |
| Upload `.txt` | File upload → chunks → index; auto-saved |
| Per-chunk delete | × button per chunk; `DELETE /documents/{id}`; auto-saved |
| Re-index corpus | Drops `corpus.txt` chunks, re-reads file, re-indexes; auto-saved |
| Persistence | Index auto-saved to `app/data/saved_index/` on every mutation |

## Next: PDF upload support

**Goal:** accept `.pdf` uploads alongside `.txt`, extract text, chunk and index the same way.

**Approach**
- Dependency: `pypdf` (pure-Python, no system libs needed)
- In `upload_file`: detect `file.filename.endswith(".pdf")`, extract text page-by-page with `pypdf.PdfReader`, join pages with `\n\n`
- Rest of the pipeline (chunk → embed → store → auto-save) unchanged
- Template: change `accept=".txt"` to `accept=".txt,.pdf"`
- Source label will show the PDF filename (e.g. `report.pdf #3`)

**Steps**
1. `uv add pypdf`
2. Update `upload_file` in `app/server.py` to branch on file extension
3. Update `accept` attribute in `app/templates/index.html`
4. Test with a real PDF

## Future ideas

- Clear index — wipe everything for a fresh start
- Query history — clickable list of recent questions (pure frontend)
- Source filtering — restrict search to a specific source via `filter=` in `similarity_search`
- Publish `turbovec` Python package to PyPI via `maturin publish`
- Benchmarks — recall vs. compression tradeoff at different `bit_width` values
