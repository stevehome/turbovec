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

## Future: adjustable chunk size

**Goal:** let the user tune `chunk_size` and `chunk_overlap` instead of the hardcoded `(500, 50)` in `_splitter`, so they can experiment with different corpora.

**Why it matters:**
- Short factual docs (Q&A, glossaries) → smaller chunks (150–250 chars) give more precise retrieval
- Long narrative docs (PDFs, articles) → larger chunks (800–1000 chars) give more context per result
- Current 500-char default is reasonable but not universal

**Constraint — ingestion-time only:** chunk size applies when text enters the index, not at query time. Changing it does not reprocess existing chunks. To see the effect the user must re-index — so the UI must make this clear. Two sensible models:
- **Settings panel with "Re-index all"** — user sets chunk_size/overlap, hits a button that drops all chunks and re-ingests every source file. Requires tracking which files were ingested (not currently stored).
- **Apply to new ingestion only** — update the splitter settings for future adds/uploads only; existing chunks are unaffected. Simpler, but the index becomes inconsistent (mixed chunk sizes).

**Recommended model:** settings panel + "Re-index all." Requires storing source file paths or raw texts alongside the index so re-ingestion doesn't require re-uploading.

**Approach**
- Add two number inputs to the corpus panel: "Chunk size" (default 500) and "Overlap" (default 50)
- Store settings in a small `settings.json` beside `saved_index/` so they persist across restarts
- `POST /settings` updates `_splitter` in place and optionally triggers re-index
- Re-index all: iterate `_store._docs`, group by source, re-chunk each source's concatenated text with the new splitter, replace chunks in the index, auto-save
- Current chunk sizes per source could be shown in the doc list header for transparency

## Deployment: Vercel vs AWS

### Why Vercel is a poor fit

Vercel runs serverless functions (Lambda under the hood). This app has three properties that clash badly with that model:

1. **Compiled Rust extension** — `turbovec` is a PyO3 `.so` built by maturin. Vercel's Python runtime doesn't support arbitrary native extensions; you'd need a custom build step and the resulting binary must match Vercel's Linux/amd64 environment. Non-trivial.
2. **Large model at startup** — `all-MiniLM-L6-v2` is ~90 MB of model weights downloaded/loaded at startup. In a serverless function this happens on every cold start (seconds of latency). There's no warm process to amortize it across requests.
3. **Persistent disk** — the index is saved to `app/data/saved_index/` (local filesystem). Serverless functions have no persistent disk between invocations; the index would be lost between cold starts without external storage.

**Verdict: skip Vercel.** The compiled extension alone rules it out without significant rework.

### AWS options (best to simplest fit)

#### Option 1: AWS App Runner (recommended for demo)

Managed container service. Runs a persistent Docker container, auto-scales, no infra to manage.

- Closest to "just works" — same process model as local dev
- Container stays warm between requests; model loads once
- No cold-start penalty for the embedding model
- Scale-to-zero is optional (min 1 instance = ~$5–15/month for a small container)

#### Option 2: AWS ECS Fargate

More control than App Runner. Better if you need fine-grained networking, IAM, or want to run alongside other ECS services.

#### Option 3: AWS EC2

Simplest conceptually — SSH in, clone repo, run the server. No autoscaling, no infra abstraction. Fine for a private demo that doesn't need to stay up.

### What would need to change for any container deployment

**1. Dockerfile (new file — biggest lift)**
```dockerfile
FROM python:3.11-slim
RUN apt-get install -y curl build-essential
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
RUN pip install maturin uv
WORKDIR /app
COPY . .
RUN cd turbovec-python && maturin build --release && uv pip install dist/*.whl
# Pre-download the embedding model so cold starts don't hit HuggingFace
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
CMD ["uv", "run", "python", "turbovec-python/app/server.py"]
```

**2. Index persistence → S3 (or accept ephemeral index)**

`_store.dump(INDEX_PATH)` writes to local disk. Two options:
- **Ephemeral**: bake a starter `saved_index/` into the Docker image at build time. Index resets on container restart, but for a demo that's acceptable.
- **Persistent**: replace `dump`/`load` calls with S3 reads/writes via `boto3`. `IdMapIndex.write()` returns bytes; upload to S3. On startup, download from S3 if the bucket key exists. Adds `boto3` dependency and an S3 bucket.

**3. ANTHROPIC_API_KEY**

Replace `.env` file with an environment variable set in the App Runner / ECS task definition. The existing `load_dotenv()` call gracefully no-ops when no `.env` is present and the env var is already set — no code change needed.

**4. Host binding**

`server.py` binds to `127.0.0.1:8000`. Change to `0.0.0.0:8000` so the container's port is reachable from outside:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

**5. Horizontal scaling (if needed)**

The app holds `_store` as a module-level global — single-process only. Multiple container replicas would each have independent indexes that diverge as users add/delete chunks. For a demo with one user this doesn't matter; for multi-user you'd need a shared index backend (e.g. S3-backed load on every write, or a dedicated index service).

### Recommended path

Start with **EC2** (fastest to validate), then move to **App Runner** once the Dockerfile is working. The S3 persistence swap is optional — a baked-in starter index is fine for a demo.

## Future ideas

- Clear index — wipe everything for a fresh start
- Query history — clickable list of recent questions (pure frontend)
- Source filtering — restrict search to a specific source via `filter=` in `similarity_search`
- Publish `turbovec` Python package to PyPI via `maturin publish`
