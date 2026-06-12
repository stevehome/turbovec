# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

turbovec is a Rust vector index with Python bindings, implementing Google Research's TurboQuant algorithm. It compresses high-dimensional vectors to 2–4 bits per coordinate and searches them with hand-written SIMD kernels (NEON on ARM, AVX-512BW/AVX2 on x86).

## Commands

### Rust

```bash
# Build
cargo build --release

# Test (release mode required — SIMD tests are meaningless in debug)
cargo test -p turbovec --release

# Single test
cargo test -p turbovec --release -- test_name

# Downstream consumer smoke test
cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml
```

On Linux, `libopenblas-dev` must be installed (`sudo apt-get install -y libopenblas-dev pkg-config`). macOS links against the Accelerate framework automatically.

### Python

```bash
# If conda is active, unset CONDA_PREFIX first (maturin conflicts with it)
unset CONDA_PREFIX

cd turbovec-python

# Install test dependencies into the uv venv
uv pip install numpy pytest

# Dev install (builds the Rust extension in-place)
uv run maturin develop --release

# Run all tests
uv run python -m pytest tests/ -v

# Single test file
uv run python -m pytest tests/test_index.py -v

# Install integration extras (required for framework integration tests)
uv pip install "langchain-core>=0.3"          # test_langchain.py
uv pip install "llama-index-core>=0.11"       # test_llama_index.py
uv pip install "haystack-ai>=2.0"             # test_haystack.py
uv pip install "agno>=2.0"                    # test_agno.py
```

## Running the web app

```bash
cd turbovec-python
uv pip install gradio sentence-transformers langchain-anthropic "langchain-core>=0.3"
uv run python app/app.py
```

Opens at http://127.0.0.1:7860. Edit `app/data/corpus.txt` to change the corpus (one document per line). Requires `ANTHROPIC_API_KEY` in the environment.

## End-to-end testing with an LLM

These approaches exercise the full RAG stack: embed documents → store in turbovec → retrieve → generate. Listed simplest-first.

### Option 1 (recommended): LangChain + Claude + local embeddings

One API key (`ANTHROPIC_API_KEY`), no embedding cost. `all-MiniLM-L6-v2` produces 384-dim vectors.

```bash
uv pip install "langchain-core>=0.3" langchain-anthropic sentence-transformers
```

The test file is at `tests/test_e2e_rag.py`. Run it with:

```bash
export ANTHROPIC_API_KEY=...
uv run python -m pytest tests/test_e2e_rag.py -v
```

Retrieval tests run without an API key; LLM tests are skipped when `ANTHROPIC_API_KEY` is not set.

### Option 2: LangChain + Claude + OpenAI embeddings

Closer to a production setup. Requires both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.

```bash
uv pip install "langchain-core>=0.3" langchain-anthropic langchain-openai
```

Replace the embeddings line with:
```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

### Option 3: Haystack pipeline + Claude

Exercises the `TurboVecDocumentStore` via Haystack's explicit pipeline abstraction.

```bash
uv pip install "haystack-ai>=2.0" "anthropic>=0.20" sentence-transformers
```

```python
from haystack import Pipeline, Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from turbovec.haystack import TurboVecDocumentStore

store = TurboVecDocumentStore()
docs = [Document(content=t) for t in [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python uses indentation to define code blocks instead of braces.",
]]

embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")
embedder.warm_up()
store.write_documents(embedder.run(docs)["documents"])

query_embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
retriever = InMemoryEmbeddingRetriever(document_store=store)

p = Pipeline()
p.add_component("embedder", query_embedder)
p.add_component("retriever", retriever)
p.connect("embedder.embedding", "retriever.query_embedding")
result = p.run({"embedder": {"text": "When was Python created?"}})
print(result["retriever"]["documents"][0].content)
```

## Workspace layout

```
turbovec/          Rust core crate (published to crates.io as `turbovec`)
turbovec-python/   PyO3 Python bindings (published to PyPI as `turbovec`)
examples/downstream-smoke/  Standalone cargo project that exercises real downstream link path
benchmarks/        Self-contained Python benchmark scripts
docs/              API reference and benchmark charts
```

## Architecture

### Rust core (`turbovec/src/`)

| Module | Role |
|--------|------|
| `lib.rs` | `TurboQuantIndex` struct; re-exports `IdMapIndex`. Constants: `BLOCK=32`, `FLUSH_EVERY=256`, `MAX_DIM=65536`. |
| `encode.rs` | Vector encoding: normalize → rotate → TQ+ calibrate → Lloyd-Max quantize → bit-pack. TQ+ calibration is fitted on the first `add` and frozen for all subsequent adds. |
| `codebook.rs` | Lloyd-Max codebook: precomputed boundaries and centroids from the Beta distribution. These are deterministic functions of `(bit_width, dim)` — no data needed. |
| `rotation.rs` | Random orthogonal rotation matrix seeded at `ROTATION_SEED=42`. Deterministic and cached via `OnceLock`. |
| `pack.rs` | SIMD-blocked repack: converts per-vector packed codes into the 32-vector interleaved layout the search kernel expects. Cached via `OnceLock`, invalidated on each `add`. |
| `search.rs` | SIMD scoring kernels: NEON (aarch64), AVX-512BW (x86 runtime-dispatched), AVX2 fallback, scalar fallback. Top-k min-heap per query. Rayon-parallelised over queries. |
| `id_map.rs` | `IdMapIndex`: stable `u64` external IDs over `TurboQuantIndex`. Bidirectional `slot ↔ id` HashMap. `remove(id)` is O(1) via `swap_remove`. |
| `io.rs` | Binary file format. `.tv` = TurboQuantIndex, `.tvim` = IdMapIndex. |
| `error.rs` | `ConstructError`, `AddError` typed errors. |

### Thread safety

`search` takes `&self` — safe to call concurrently. Lazy caches (`rotation`, `boundaries`, `centroids`, `blocked`) are `OnceLock<_>` and initialised once by the first caller. `add` takes `&mut self` and resets `blocked` by replacing its `OnceLock`. Call `prepare()` after loading or batch-adding to warm the caches before serving queries.

### Python layer (`turbovec-python/`)

- `src/lib.rs` — PyO3 bindings. Wraps `TurboQuantIndex` and `IdMapIndex`, converts numpy arrays to `&[f32]`, maps Rust errors to Python `ValueError`/`RuntimeError`.
- `python/turbovec/__init__.py` — re-exports both index types from the native extension.
- `python/turbovec/{langchain,llama_index,haystack,agno}.py` — drop-in replacements for each framework's in-tree reference vector store.

### SIMD dispatch

- **ARM**: NEON always used. No runtime dispatch.
- **x86**: `.cargo/config.toml` sets `target-cpu=x86-64-v3` (AVX2 baseline). AVX-512BW kernel is selected at runtime via `is_x86_feature_detected!`. The `FORCE_SCALAR_FALLBACK` atomic in `search.rs` exists only under `#[cfg(test)]` to exercise the scalar path.

### Key constraints

- `dim` must be a positive multiple of 8.
- `bit_width` must be in `{2, 3, 4}`.
- 64-bit targets only; the crate emits `compile_error!` on 32-bit.
- Input coordinates must be finite and `|value| < 1e16`; violating this panics in `add`/`search` (returns `AddError::InvalidInputValue` from `add_2d`).
- The crate is 64-bit by design — `usize` arithmetic in encode/pack/search assumes a 64-bit address space.
