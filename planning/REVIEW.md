# Project Review

## Structure

```
turbovec/                    Rust core crate (crates.io: turbovec)
  src/{lib,encode,search,pack,codebook,rotation,id_map,io,error}.rs
  tests/                     15 integration test files
  Cargo.toml

turbovec-python/             PyO3 wheel (PyPI: turbovec)
  src/lib.rs                 PyO3 glue — wraps Rust types, converts numpy arrays
  python/turbovec/
    __init__.py              re-exports TurboQuantIndex, IdMapIndex
    langchain.py  (558 ln)   LangChain VectorStore adapter
    llama_index.py           LlamaIndex adapter
    haystack.py              Haystack adapter
    agno.py                  Agno adapter
    _dedup.py / _persist.py  helpers
  app/                       ← demo app (NOT in the wheel)
    server.py    (505 ln)
    templates/index.html
    data/
  tests/                     Python integration tests
  pyproject.toml

examples/downstream-smoke/   standalone Cargo project, tests real downstream linking
benchmarks/suite/            self-contained Python benchmark scripts
docs/                        API docs, SVG charts, test PDFs
planning/                    PLAN.md, REVIEW.md
Dockerfile                   multi-stage; builder=Rust+maturin, runtime=Python+model
```

---

## Issues

**1. `pyproject.toml` ships server-only deps as hard requirements**

`psutil`, `pypdf`, `python-dotenv`, and `langchain-text-splitters` are listed under
`[project] dependencies` but are only used by `app/server.py` — not by the package itself.
Anyone who does `pip install turbovec` gets all four pulled in unnecessarily.

**2. `server.py` is 505 lines of mixed concerns**

Routes, OCR, enrichment, chunking, persistence, and HTML templating are all in one file.
Natural split: `ingest.py` (chunking / OCR / enrichment), routes stay in `server.py`.

**3. Global mutable state**

`_store`, `_sources`, `_chunk_size`, etc. are module-level globals mutated by multiple
endpoints. Safe with one Uvicorn worker, breaks silently under `--workers 2`.

**4. `_extract_text` / `_extract_text_with_claude` are synchronous**

They block the asyncio event loop during large PDF processing. Should be wrapped in
`loop.run_in_executor(None, ...)`.

**5. `sources.json` holds full text of every uploaded document**

A 10 MB PDF produces a 10 MB `sources.json`. Better to store each source as a separate
file under `data/sources/<filename>`.

**6. `plan/` and `planning/` both exist** — one directory is enough.

**7. `docs/` has personal test PDFs in git** — excluded from Docker but committed to the repo.

**8. No authentication** — if deployed publicly anyone can upload, delete, or clear the index.

---

## Clean separation — shipping without source

The goal: publish `turbovec` to crates.io and PyPI, then run the demo app by installing
from those registries, no source needed.

**Step 1 — Fix `pyproject.toml`**

Move server-only deps out of `[project] dependencies` into an `[app]` optional group:

```toml
[project]
dependencies = ["numpy>=1.20"]   # only true runtime dep of the package

[project.optional-dependencies]
langchain   = ["langchain-core>=0.3"]
llama-index = ["llama-index-core>=0.11"]
haystack    = ["haystack-ai>=2.0"]
agno        = ["agno>=2.0"]
app         = [
    "fastapi", "uvicorn", "jinja2", "python-multipart",
    "sentence-transformers", "langchain-anthropic",
    "langchain-core>=0.3", "langchain-text-splitters",
    "pypdf", "psutil", "python-dotenv", "anthropic",
]
```

**Step 2 — Move the demo app out of `turbovec-python/`**

```
turbovec-demo/        ← new top-level directory (or separate repo)
  pyproject.toml      depends on: turbovec[app]
  server.py
  templates/
  data/
  Dockerfile
  README.md
```

`turbovec-python/` then contains only the wheel — bindings + adapters, no app code.

**Step 3 — Publish**

```bash
# Rust crate (Ryan's credentials)
cargo publish -p turbovec

# Python wheel (Ryan's credentials)
cd turbovec-python && maturin publish
```

**Step 4 — Users run the demo with no source**

```bash
pip install "turbovec[app]"
# download server.py + templates/ from turbovec-demo repo
export ANTHROPIC_API_KEY=sk-...
uvicorn server:app --host 0.0.0.0 --port 8000
```

Or Docker (already works today):

```bash
docker build -t turbovec-rag .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... turbovec-rag
# opens at http://localhost:8000
```

---

## User instructions (current state, source required)

```bash
# 1. Clone
git clone https://github.com/stevehome/turbovec-demo
cd turbovec-demo

# 2. Build the Rust extension
cd turbovec-python
unset CONDA_PREFIX          # if conda is active
uv run maturin develop --release

# 3. Install app dependencies
uv pip install fastapi uvicorn jinja2 python-multipart \
    sentence-transformers langchain-anthropic "langchain-core>=0.3" \
    langchain-text-splitters pypdf psutil python-dotenv anthropic

# 4. Set API key
echo "ANTHROPIC_API_KEY=sk-..." > .env

# 5. Run
uv run python app/server.py
# opens at http://127.0.0.1:8000
```

---

## Previous review (Phase 1 → Phase 2 transition)

See `plan/REVIEW.md` for the earlier review written before the FastAPI + HTMX app was
built — covers Gradio Phase 1 gaps and the professional frontend recommendation that led
to the current HTMX implementation.
