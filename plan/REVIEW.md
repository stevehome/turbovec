# Project Review & Professional Frontend Suggestions

## What's working well

### Core library
The Rust core is the strongest part of the project. The SIMD kernels, the TurboQuant algorithm implementation, and the file format are production-quality. The test suite is thorough — 12 integration test files covering codebook correctness, concurrent search, input validation, IO versioning, TQ+ calibration, and more. The security hardening (MAX_DIM bounds, NaN/Inf guards, untrusted-load checks) is well considered.

### Python bindings
The PyO3 layer is clean and idiomatic. `TurboQuantVectorStore` correctly implements the LangChain `VectorStore` interface including async paths, filter support, persistence, and the `embeddings` property. The dedup and upsert semantics match the `InMemoryVectorStore` reference behaviour, which is exactly the right bar for a drop-in replacement.

### Framework integrations
All four integrations (LangChain, LlamaIndex, Haystack, Agno) exist and are tested. The pattern of structurally comparing against the canonical in-tree reference store is the right approach.

### Gradio app (Phase 1)
Functional as a demo tool. Loads a corpus, indexes it, answers questions with Claude. Sufficient for internal use and for showing the project off.

---

## What the current app is missing for professional use

| Gap | Impact |
|-----|--------|
| Single in-memory index — restart loses everything | Can't persist or grow the corpus across sessions |
| No corpus management | Can't add, edit, or delete documents from the UI |
| No file upload | Users must edit `corpus.txt` manually |
| No streaming LLM output | Answers appear all at once after a delay |
| No authentication | Open to anyone on the network |
| Single-user — no session isolation | One user's queries affect another's state |
| Gradio look-and-feel | Hard to brand or customise beyond basic CSS |

---

## Professional frontend recommendation

### Architecture

Split into two processes with a clean boundary:

```
┌─────────────────────┐       ┌──────────────────────────────┐
│   Next.js frontend  │ HTTP  │   FastAPI backend             │
│   (React + Tailwind)│ ───── │   /api/query  (POST)         │
│   localhost:3000    │       │   /api/documents  (GET/POST) │
└─────────────────────┘       │   /api/upload  (POST)        │
                               │   localhost:8000             │
                               └──────────────────────────────┘
```

**Backend — FastAPI**
- One `TurboQuantVectorStore` per session (stored server-side by session token), or a shared index for a single-user deployment.
- `/api/query` — takes `{question, k}`, returns `{answer, sources}` as a **streaming** response (Server-Sent Events) so the answer appears word by word.
- `/api/documents` — GET returns the current document list; POST adds a document.
- `/api/upload` — accepts a `.txt` file, splits on newlines, adds all lines to the index.
- `/api/index/save` and `/api/index/load` — persistence via `store.dump()` / `TurboQuantVectorStore.load()`.

**Frontend — Next.js + Tailwind**
- Three panels: document list (left), query + answer (centre), settings (right).
- Answer streams in via `EventSource` — no waiting for the full response.
- File drag-and-drop for corpus upload.
- Toast notifications on add/delete.

### Minimal file layout

```
app/
  backend/
    main.py          # FastAPI app
    store.py         # index lifecycle, session management
    stream.py        # SSE streaming helper for Claude
  frontend/
    app/
      page.tsx       # main layout
      components/
        QueryBox.tsx
        AnswerStream.tsx
        DocumentList.tsx
        UploadZone.tsx
    tailwind.config.ts
    package.json
```

### Simpler alternative: FastAPI + HTMX

If a separate Node/npm build step is too much overhead, FastAPI + HTMX achieves 80% of the above with zero JavaScript build tooling:

- Jinja2 templates rendered server-side.
- HTMX attributes handle the dynamic parts (streaming answer, document list refresh, upload).
- No React, no npm, no bundler.
- Answer streaming via HTMX `hx-swap="beforeend"` on a chunked response.

This is the right choice if the app is internal / demo-only. Choose Next.js if the app will be customer-facing or needs a polished design system.

### Recommended stack (external / customer-facing)

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + uvicorn | Async, streaming, easy SSE, matches uv ecosystem |
| Frontend | Next.js 14 (App Router) | SSR, fast, broad component ecosystem |
| Styling | Tailwind + shadcn/ui | Professional components with zero custom CSS |
| Streaming | Server-Sent Events | Simple, no WebSocket overhead for one-way text |
| Auth | NextAuth.js | Easy OAuth / magic-link with minimal setup |
| Persistence | `store.dump()` to local disk or S3 | Already supported by turbovec |

### Recommended stack (internal / demo)

| Layer | Choice |
|-------|--------|
| Backend | FastAPI + uvicorn |
| Frontend | HTMX + Jinja2 templates |
| Styling | Pico CSS or MVP.css (classless, no build step) |
| Streaming | chunked HTTP / SSE |

---

## Suggested implementation order for Phase 2

1. **FastAPI backend** — wrap the existing `app.py` logic into `/api/query` (streaming) and `/api/documents` endpoints. ~100 lines.
2. **Streaming answer** — replace `llm.invoke` with `llm.stream` and emit tokens as SSE. Biggest UX win for lowest effort.
3. **Document add + upload** — POST endpoint + simple form or drag-and-drop. Enables live corpus growth.
4. **Persistence** — auto-save `store.dump()` after each add; load on startup if a saved index exists.
5. **React frontend** (optional) — swap Jinja2 for Next.js once the API is stable.
