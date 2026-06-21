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
| Adjustable chunk size | `POST /rechunk` with `chunk_size`/`chunk_overlap`; re-chunks all tracked sources; persisted to `settings.json`/`sources.json` |

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

Three numbers, worth showing together:

1. **Quantized index size** — the packed codes actually held in memory. Reuse the benchmark technique: write the index to a temp path via `_store._index.write()` and check file size. Exact, cheap at demo scale, and directly comparable to the compression benchmark table.
2. **FP32 equivalent size** — what the same vectors would cost uncompressed: `n_vectors × dim × 4 bytes`. Both `n` and `dim` are available from `len(_store._docs)` and `_store._index.dim`. No I/O needed — pure arithmetic.
3. **Process RSS** — total server memory (embedding model + torch + index + FastAPI). Dominated by `sentence-transformers` (~100s of MB); won't move much with `bit_width` changes but answers "is this heavy to run." Use `psutil.Process().memory_info().rss` rather than stdlib `resource.getrusage` (peak not current; platform-inconsistent units).

Showing all three together makes the compression ratio live and tangible:

```
Vectors: 2.1 MB quantized · 73.6 MB uncompressed (35x) · Process: 412 MB
```

The compression ratio from the live index should match the benchmark table — a useful sanity check.

**How to compute quantized size without disk I/O (alternative)**

The packed code size is deterministic: `n_vectors × dim × bit_width / 8` bytes, plus a small fixed header. This avoids the write-to-temp-file step entirely and is instant. The file-size approach is more honest (captures actual overhead), but the formula is good enough for a UI label.

**Approach**
- Add a stats line in the corpus panel below the chunk count
- New `GET /stats` endpoint returning `{quantized_mb, fp32_mb, ratio, process_mb}` as JSON
- HTMX `hx-get="/stats"` div that refreshes alongside the doc list after every mutation
- `uv add psutil` for process RSS

## ~~Future: adjustable chunk size~~ (shipped)

> Shipped in commit `a47b818`. See features table above.

**Goal:** let the user tune `chunk_size` and `chunk_overlap` instead of the hardcoded `(500, 50)` in `_splitter`, so they can experiment with different corpora.

**Why it matters:**
- Short factual docs (Q&A, glossaries) → smaller chunks (150–250 chars) give more precise retrieval
- Long narrative docs (PDFs, articles) → larger chunks (800–1000 chars) give more context per result
- Current 500-char default is reasonable but not universal

**Constraint — ingestion-time only:** chunk size applies when text enters the index, not at query time. Changing it does not reprocess existing chunks. To see the effect the user must re-index — so the UI must make this clear. Two sensible models:
- **Settings panel with "Re-index all"** — user sets chunk_size/overlap, hits a button that drops all chunks and re-ingests every source file. Requires tracking which files were ingested (not currently stored).
- **Apply to new ingestion only** — update the splitter settings for future adds/uploads only; existing chunks are unaffected. Simpler, but the index becomes inconsistent (mixed chunk sizes).

**Recommended model:** settings panel + "Re-index all." Requires storing the original full text per source so re-chunking is lossless.

**The overlap reconstruction problem:** joining existing chunks to recover original text is lossy when `chunk_overlap > 0` (overlap regions appear twice). The clean fix is a `_sources: dict[str, str]` global that stores the full original text keyed by source name, populated at upload/add time. Re-chunking then reads from `_sources` rather than trying to reconstruct from existing chunks.

**Approach**
1. Add `_sources: dict[str, str]` global — store full text per source on every upload/add; persist alongside `saved_index/` as `sources.json`
2. Add two number inputs to the corpus panel: "Chunk size" (default 500) and "Overlap" (default 50)
3. `POST /settings` — updates `_splitter`, re-chunks all sources from `_sources`, rebuilds index, auto-saves
4. Store current settings in `settings.json` beside `saved_index/` so they survive restarts

## Future: delete by source

**Goal:** remove all chunks from a specific source (e.g. delete everything from `git-usermanual.txt`) without deleting chunks from other sources one by one.

**Backend:** already supported — same pattern as the reindex endpoint:
```python
old_ids = [sid for sid, (_, meta) in _store._docs.items() if meta.get("source") == name]
_store.delete(old_ids)
```
Just needs a `DELETE /sources/{name}` endpoint.

**UI options (best to simplest):**

- **Option A: Grouped doc list with per-source delete (recommended)** — restructure `_doc_list_html()` to group chunks by source name, render a collapsible `<details>` per source with a delete-source × button in the summary. Makes the list much easier to scan at scale (e.g. 200 chunks from 3 files). The per-chunk × buttons remain inside each group.
- **Option B: Dropdown + delete button** — a `<select>` of unique source names + "Delete source" button. Simpler to implement, no restructuring.
- **Option C: Clickable source label** — each `corpus.txt #1` label deletes that whole source on click. Discoverable but risks accidental deletion.

**Recommended:** Option A — grouping by source is independently useful regardless of deletion, and the delete button comes along naturally. Pairs well with the `_sources` dict from the chunk-size feature (both need per-source tracking).

**Steps**
1. Add `DELETE /sources/{name}` endpoint
2. Restructure `_doc_list_html()` to group by `meta["source"]`, render `<details>` per group
3. Add delete-source button in each group's `<summary>`
4. Auto-save after deletion

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

## Future: pre-processing before chunking

Right now text enters the index raw — directly from the textarea, uploaded file, or PDF extraction. Three approaches worth considering, ordered by impact:

### 1. Contextual Retrieval (highest impact, most aligned with our stack)

Anthropic's own technique: before indexing each chunk, call Claude to prepend a one-sentence context explaining the chunk's place in the source document.

```
"This passage is from the Python PEP-8 style guide and describes naming conventions for variables and functions."

<original chunk text>
```

The context is prepended to the chunk *before* embedding, so the vector captures both the specific content and its broader meaning. Retrieval improves dramatically for chunks that lose meaning in isolation (e.g. "It was introduced in version 3.10" — introduced *what*?).

**Cost:** one LLM call per chunk at index time. At demo scale (~100 chunks from a PDF) that's cheap and fast. Use `claude-haiku-4-5` (already a dep) with a small prompt.

**Implementation:** wrap `_chunk_with_meta` to optionally call Claude for each chunk, prepend the context, then embed the combined text. Store the original text in `_docs` (for display) but embed the enriched version.

**Why it matters more here than for most RAG demos:** turbovec uses highly compressed 2–4 bit vectors. A chunk that *clearly* expresses its own meaning compresses better — the quantization noise matters less when the signal is strong. Contextual enrichment and quantization are complementary.

### 2. PDF noise stripping (practical, low effort)

`pypdf.PdfReader` extracts text faithfully but PDFs commonly contain:
- Repeated headers/footers ("Page 3 of 47", company name)
- Table of contents / index pages
- Watermarks ("DRAFT", "CONFIDENTIAL") that pollute every chunk

A simple regex pass after extraction removes the worst offenders:
```python
import re
text = re.sub(r'(?m)^\s*Page \d+ of \d+\s*$', '', text)
text = re.sub(r'\n{3,}', '\n\n', text)  # collapse excessive blank lines
```

For heavier PDFs, `pymupdf` (fitz) extracts text with layout awareness and can skip header/footer regions by bounding box.

**Worth doing** any time PDF upload is a primary use case. Low risk, no LLM cost.

### 3. Semantic chunking (better boundaries, higher complexity)

LangChain's `SemanticChunker` splits at embedding similarity drops rather than character count — chunks align with topic shifts rather than arbitrary size limits. Better for long narrative documents (legal, academic).

**Tradeoff:** requires N+1 embedding calls during ingestion (one per candidate split point) rather than zero. For a small corpus that's fine; at scale it adds latency. Also makes chunk size unpredictable, which is harder to explain in the UI.

**Skip for now unless** users complain that the fixed-size chunks cut across sentence or paragraph boundaries mid-thought.

### What NOT to do: stop-word / noise-word removal

Removing "the", "a", "and" etc. before chunking is a BM25-era technique for sparse retrieval. Dense embeddings (sentence-transformers) already handle function words internally — stripping them can hurt rather than help by degrading the sentence-level representations the model was trained on. Don't do this.

### Recommended next step

**Contextual Retrieval** — highest ROI, uses Claude which is already wired in, and directly improves the quality of what turbovec is compressing. Add a toggle in the upload flow: "Enrich chunks with context (uses Claude, slower)" — off by default so the fast path stays fast.

## Future: scanned PDF ingestion

Current `pypdf.PdfReader` extracts the text layer only — it returns an empty string for image-based (scanned) PDFs. A scanned annual report or contract uploads silently with zero chunks. Three approaches, from simplest to most involved:

### Approach A: Claude native PDF API (recommended)

Borrowed from [Anthropic's platform PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support). Instead of parsing with pypdf, send the PDF directly to Claude as a `document` block:

```python
import base64, anthropic

client = anthropic.Anthropic()
with open("scanned.pdf", "rb") as f:
    pdf_b64 = base64.standard_b64encode(f.read()).decode()

msg = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_b64}},
            {"type": "text", "text": "Extract all text from this document, preserving paragraph structure."}
        ]
    }]
)
text = msg.content[0].text
```

Claude converts each page to an image internally, runs its own OCR, and returns clean text — including tables, diagrams with captions, and mixed text/image pages. Works for both text-layer PDFs and fully scanned ones. We already depend on `langchain-anthropic`, so no new dependencies needed.

**Limits:** 32 MB per request, 100 pages per request (200k context models like haiku). **Cost:** ~1,500–3,000 text tokens + ~1,000 image tokens per page. A 20-page scanned report ≈ ~50k tokens ≈ $0.025 at haiku pricing — cheap enough to do on upload.

**Prompt caching opportunity:** For large PDFs that need multiple questions (e.g. extract text, then enrich), upload once to the Files API and reference by `file_id` to avoid re-sending the binary on each call.

### Approach B: Anthropic skills repo approach (pdf2image + pytesseract)

The [anthropics/skills](https://github.com/anthropics/skills) PDF skill uses traditional OCR:

```python
# uv add pdf2image pytesseract
# also needs: brew install poppler tesseract
from pdf2image import convert_from_path
import pytesseract

images = convert_from_path("scanned.pdf")
text = "\n\n".join(pytesseract.image_to_string(img) for img in images)
```

**Pros:** free, local, no API cost, no data leaves the machine.  
**Cons:** tesseract quality is noticeably worse than Claude on anything but clean typefaces; needs `poppler` and `tesseract` system dependencies (complicates the Dockerfile); poor on tables, handwriting, and rotated pages.

Good fit if the corpus is high-volume and privacy is paramount. Not the right choice for a demo that already uses Claude.

### Approach C: Page-image rendering + Claude vision

Render pages individually with `pymupdf` (fitz) for fine-grained control, then send each page image to Claude:

```python
# uv add pymupdf
import fitz  # pymupdf
doc = fitz.open("scanned.pdf")
for page in doc:
    pix = page.get_pixmap(dpi=150)
    png_bytes = pix.tobytes("png")
    # send png_bytes as image/png to Claude messages API
```

Useful when you need control over rendering resolution, want to skip certain page ranges, or need to extract structured data (tables) from specific pages. More code than Approach A for the same outcome at demo scale.

### Recommended implementation plan

1. **Auto-detect** in `_extract_text`: after pypdf extraction, check if text is suspiciously sparse (`len(text.strip()) < 100 * num_pages`). If so, fall back to Approach A.
2. **No new dependency** — `anthropic` is already available via `langchain-anthropic`.
3. **Combine with contextual enrichment**: if `_contextual` is on, pass the raw extracted text into `_enrich_chunks` as usual — extraction and enrichment are independent steps.
4. **Dockerfile**: no changes needed for Approach A (no system deps). Approach B would need `apt-get install -y tesseract-ocr poppler-utils`.

```python
def _extract_text(filename: str, data: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(io.BytesIO(data))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        # Fall back to Claude OCR if text layer is empty/sparse
        if len(text.strip()) < 100 * len(reader.pages):
            text = _extract_text_with_claude(data)
        return text
    return data.decode(errors="replace")


def _extract_text_with_claude(pdf_bytes: bytes) -> str:
    import anthropic, base64
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "document", "source": {
                    "type": "base64", "media_type": "application/pdf",
                    "data": base64.standard_b64encode(pdf_bytes).decode()
                }},
                {"type": "text", "text": "Extract all text from this document, preserving paragraph structure. Output only the extracted text."}
            ]
        }]
    )
    return msg.content[0].text
```

## Future: source document viewer

After a query the Sources panel shows chunk previews and scores — but if a user wants to see more of a retrieved passage, or verify the context around it, there's no way to do that without going back to the original file.

### What to show

The goal is to let the user click a source reference and see the full original text of that source, with the retrieved chunk highlighted. Three levels of detail:

1. **Full source text in a modal** — on click, open a `<dialog>` element containing the full text from `_sources[source_name]`, with the matched chunk highlighted via `<mark>`. Simple, works for all source types, no extra storage.

2. **Surrounding context only** — instead of the full document, show N characters before and after the chunk boundary. Gives the "read more" experience without overwhelming the modal with a 50-page PDF. Good default.

3. **Page-level view for PDFs** — for uploaded PDFs, re-render the specific page as an image via `pymupdf` and show it alongside the extracted text. Best experience for scanned or visually complex documents, but adds `pymupdf` dependency and per-request rendering cost.

### Recommended approach

Level 2 + level 1 toggle: default to ±500 chars of surrounding context (fast, always available from `_sources`), with a "Show full document" link that expands to the full text. Highlight the chunk with `<mark>`.

**Implementation sketch:**

New `GET /sources/{source_name}/context?chunk_index=N&window=500` endpoint:
```python
@app.get("/sources/{source_name}/context")
async def source_context(source_name: str, chunk_index: int = 0, window: int = 500):
    full_text = _sources.get(source_name, "")
    # Find the chunk text in the full source
    chunk_text = next(
        (text for _, (text, meta) in _store._docs.items()
         if meta.get("source") == source_name and meta.get("chunk") == chunk_index),
        None
    )
    if not chunk_text or not full_text:
        return JSONResponse({"error": "not found"}, status_code=404)
    pos = full_text.find(chunk_text[:80])  # find by first 80 chars (avoids context-enrichment prefix mismatch)
    start = max(0, pos - window)
    end = min(len(full_text), pos + len(chunk_text) + window)
    return JSONResponse({
        "before": full_text[start:pos],
        "chunk": chunk_text,
        "after": full_text[pos + len(chunk_text):end],
        "truncated_start": start > 0,
        "truncated_end": end < len(full_text),
    })
```

The Sources panel renders each result with a small "view in context ↗" link. On click, HTMX fetches the endpoint and populates a `<dialog>`:
```html
<dialog id="context-modal">
  <article>
    <header><strong id="ctx-source"></strong> <button onclick="this.closest('dialog').close()">×</button></header>
    <p id="ctx-before" style="color:var(--pico-muted-color)"></p>
    <mark id="ctx-chunk"></mark>
    <p id="ctx-after" style="color:var(--pico-muted-color)"></p>
  </article>
</dialog>
```

**Wrinkle — contextual enrichment:** when `_contextual` is on, chunks in `_store._docs` have the Claude-prepended context sentence at the front. The `find()` lookup needs to strip that prefix before searching. One clean fix: store the original (pre-enrichment) chunk text separately in metadata as `{"source": …, "chunk": …, "original": …}` and look up by `meta["original"]` instead of `text`.

**Why it matters:** without source viewing, users have to trust the chunk preview blind. Being able to read the surrounding passage builds confidence in the retrieval quality — especially useful when demoing the bit-width tradeoff (lower bit-width may retrieve adjacent but not exactly right chunks).

---

## Future: rate limit handling for contextual enrichment

The current `_enrich_chunks` fires all N chunk enrichments concurrently via `asyncio.gather`. For a large upload (e.g. a 200-chunk PDF), that's 200 simultaneous haiku calls — well over the 500k input tokens/minute org rate limit, which we already hit during testing.

### The problem in numbers

A typical chunk is ~500 chars ≈ ~125 tokens. The contextual enrichment prompt includes the full source document as context. For a 10-page PDF (~5,000 chars ≈ 1,250 tokens) with 20 chunks:
- Tokens per call: ~1,250 (document) + ~125 (chunk) + ~50 (prompt) ≈ ~1,425 input tokens
- 20 concurrent calls: ~28,500 tokens fired simultaneously
- For 200 chunks: ~285,000 tokens in one burst → rate limit guaranteed

### Fix: semaphore-based concurrency cap

The simplest fix — cap concurrent Claude calls with `asyncio.Semaphore`:

```python
_ENRICHMENT_CONCURRENCY = 5  # tune based on rate limit tier

async def _enrich_chunks(chunks: list[str], source_text: str) -> list[str]:
    sem = asyncio.Semaphore(_ENRICHMENT_CONCURRENCY)

    async def _one(chunk: str) -> str:
        async with sem:
            prompt = (
                f"<document>\n{source_text}\n</document>\n\n"
                f"<chunk>\n{chunk}\n</chunk>\n\n"
                "Give a short succinct context to situate this chunk within the overall document "
                "for the purposes of improving search retrieval. Answer only with the succinct context and nothing else."
            )
            resp = await _llm.ainvoke([HumanMessage(content=prompt)])
            return f"{resp.content.strip()}\n\n{chunk}"

    return list(await asyncio.gather(*[_one(c) for c in chunks]))
```

5 concurrent calls is conservative and safe at all tiers. For tier-2+ accounts (500k TPM sustained), 10–20 is fine. Make it a constant at the top of server.py so it's easy to adjust.

### Additional improvement: prompt caching on the source document

The document context is the same for every chunk enrichment call from the same source. Anthropic's prompt caching (`cache_control: {type: ephemeral}`) caches the document prefix across calls with a 5-minute TTL — reduces input token cost by ~90% after the first call and also reduces latency.

To use it with LangChain, pass the cache control via extra headers or switch the enrichment calls to the raw `anthropic` SDK (already imported as `_anthropic_sdk`) where `cache_control` is a first-class parameter:

```python
async def _enrich_one(chunk: str, source_text: str, client: _anthropic_sdk.AsyncAnthropic) -> str:
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"<document>\n{source_text}\n</document>",
                 "cache_control": {"type": "ephemeral"}},  # cached across chunk calls
                {"type": "text", "text": f"<chunk>\n{chunk}\n</chunk>\n\nGive a short succinct context..."},
            ]
        }]
    )
    return f"{msg.content[0].text.strip()}\n\n{chunk}"
```

Combined with the semaphore: 5 concurrent calls, first call pays full token cost, subsequent calls hit the cache. For a 200-chunk document, that's ~199 cache hits — cost drops by ~10x and rate limit pressure drops by the same factor.

### Recommended implementation

1. Add `_ENRICHMENT_CONCURRENCY = 5` constant
2. Wrap `_enrich_chunks` with `asyncio.Semaphore(_ENRICHMENT_CONCURRENCY)`
3. Switch enrichment calls from `_llm.ainvoke` to raw `_anthropic_sdk.AsyncAnthropic` with `cache_control` on the document block
4. Set `max_tokens=200` for enrichment (context sentences are short — no need for 1024)

---

## Future ideas

- Clear index — wipe everything for a fresh start
- Query history — clickable list of recent questions (pure frontend)
- Source filtering — restrict search to a specific source via `filter=` in `similarity_search`
- Publish `turbovec` Python package to PyPI via `maturin publish`
