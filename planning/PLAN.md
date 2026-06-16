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
| PDF upload | `.pdf` accepted alongside `.txt`; text extracted via `pypdf.PdfReader` |

## Benchmarks (done)

Synthetic benchmarks (no dataset download needed) cover `bit_width` in `{2, 3, 4}`:

| bit_width | compression vs FP32 | recall@1 | recall@4 | recall@16 |
|-----------|---------------------|----------|----------|-----------|
| 2-bit | ~16x | ~0.37 | ~0.68 | ~0.91 |
| 3-bit | ~10x | ~0.61 | ~0.91 | ~0.996 |
| 4-bit | ~8x | ~0.78 | ~0.98 | ~1.00 |

3-bit is the standout middle ground: most of 4-bit's recall at close to 2-bit's compression. This motivates exposing `bit_width` as a user choice (see below) rather than hardcoding 4-bit.

Scripts: `benchmarks/suite/{compression,recall}_synthetic.py`. Results: `benchmarks/results/{compression,recall}_synthetic.json`.

## Next: bit_width selector in the UI

**Goal:** let the user choose 2/3/4-bit instead of the hardcoded `bit_width=4` in `server.py`, informed by the table above.

**Constraint:** `bit_width` is fixed at index construction — TQ+ calibration freezes on the first `add` (see `encode.rs`). It cannot be changed on a live index; changing it means **rebuilding from scratch**: re-create the `TurboQuantVectorStore` with the new `bit_width`, then re-add every existing `(text, metadata)` pair already held in `_store._docs` (no need to re-read source files — the text is already in memory). This is a rebuild operation, not a live toggle, and should be presented that way (similar to "Re-index corpus.txt", with a confirm prompt).

**Approach**
- Add a `<select>` in the corpus panel: "Compression: 2-bit (~16x) / 3-bit (~10x) / 4-bit (~8x, default)"
- New endpoint `POST /rebuild` with `bit_width` form field:
  - Collect `[(text, meta) for text, meta in _store._docs.values()]` and ids
  - Build a fresh `TurboQuantVectorStore(embedding=_embeddings, bit_width=new_bit_width)`
  - `new_store.add_texts(texts, metadatas=metas, ids=ids)`, swap `_store` global, `dump(INDEX_PATH)`
- Show the active bit_width next to the chunk count, e.g. "9 chunks in index (4-bit)"
- Persist the chosen bit_width across restarts — `TurboQuantVectorStore.load` already round-trips `bit_width` from `docstore.json`, so this is automatic once rebuilt and saved

**Open question:** rebuild re-embeds nothing (embeddings are cached as text), but it does re-quantize all vectors — for the demo-scale corpus (tens of chunks) this is instant; not a concern at this scale.

## Next: memory usage monitoring

**Goal:** surface how much memory the running app is actually using, so the compression benchmark numbers connect to something visible in the live demo.

Two distinct numbers, worth showing separately:

1. **Index footprint** — the quantized vector data only, which is what `bit_width` actually affects. Reuse the benchmark technique (`benchmarks/suite/compression_synthetic.py`): write the index to a temp path via `index.write()` and check file size. Exact, and ties directly to the compression table above. Cheap at demo scale (instant for tens of MB).
2. **Process RSS** — total memory of the running server (embedding model + torch + index + FastAPI). Dominated by the `sentence-transformers` model (~100s of MB), so this number won't move much when `bit_width` changes — it mainly answers "is this app heavy to run locally." Use `psutil.Process().memory_info().rss` (one new dependency) rather than stdlib `resource.getrusage` (peak, not current; platform-inconsistent units).

**Approach**
- Add a small stats line in the corpus panel: "Index: 2.1 MB · Process: 412 MB"
- New endpoint `GET /stats` returning both numbers as JSON; doc-list `hx-get` already polls `/documents` on load, could add a sibling `hx-get="/stats"` div, or fold both into the existing `/documents` response
- `uv add psutil` for the process RSS number

## Future ideas

- Clear index — wipe everything for a fresh start
- Query history — clickable list of recent questions (pure frontend)
- Source filtering — restrict search to a specific source via `filter=` in `similarity_search`
- Publish `turbovec` Python package to PyPI via `maturin publish`
