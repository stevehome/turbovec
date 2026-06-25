"""Global application state, path constants, and startup initialisation."""
from __future__ import annotations

import json
import shutil
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer

from turbovec.langchain import TurboQuantVectorStore

_HERE = Path(__file__).parent
CORPUS_PATH   = _HERE / "data" / "corpus.txt"
INDEX_PATH    = _HERE / "data" / "saved_index"
SETTINGS_PATH = _HERE / "data" / "settings.json"
SOURCES_DIR   = _HERE / "data" / "sources"
OCR_DIR       = _HERE / "data" / "ocr"
SOURCES_DIR.mkdir(exist_ok=True)
OCR_DIR.mkdir(exist_ok=True)
K = 3

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# BGE retrieval models expect this prefix on queries (not on documents).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(_BGE_QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def load_settings() -> tuple[int, int, bool]:
    if SETTINGS_PATH.exists():
        s = json.loads(SETTINGS_PATH.read_text())
        return int(s.get("chunk_size", 500)), int(s.get("chunk_overlap", 50)), bool(s.get("contextual", False))
    return 500, 50, False


def save_settings(chunk_size: int, chunk_overlap: int, contextual: bool) -> None:
    SETTINGS_PATH.write_text(json.dumps({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "contextual": contextual}))


def load_sources() -> dict[str, str]:
    # Migrate from legacy sources.json if present.
    legacy = _HERE / "data" / "sources.json"
    if legacy.exists():
        data = json.loads(legacy.read_text())
        for name, text in data.items():
            (SOURCES_DIR / name).write_text(text)
        legacy.unlink()
        print(f"Migrated {len(data)} sources from sources.json → data/sources/")
    return {p.name: p.read_text() for p in sorted(SOURCES_DIR.iterdir()) if p.is_file()}


def save_source(name: str, text: str) -> None:
    """Write or overwrite a single source file."""
    (SOURCES_DIR / name).write_text(text)


def delete_source_file(name: str) -> None:
    """Remove a single source file (no-op if missing)."""
    (SOURCES_DIR / name).unlink(missing_ok=True)


def save_sources() -> None:
    """Rewrite all source files to match state.sources (used after bulk ops)."""
    existing = {p.name for p in SOURCES_DIR.iterdir() if p.is_file()}
    for name, text in state.sources.items():
        (SOURCES_DIR / name).write_text(text)
    for orphan in existing - state.sources.keys():
        (SOURCES_DIR / orphan).unlink(missing_ok=True)


def chunk_with_meta(text: str, source: str, visibility: str = "shared") -> tuple[list[str], list[dict]]:
    chunks = state.splitter.split_text(text)
    metas = [{"source": source, "chunk": i, "visibility": visibility} for i, _ in enumerate(chunks)]
    return chunks, metas


def _build_index_from_corpus() -> TurboQuantVectorStore:
    print(f"Indexing corpus from {CORPUS_PATH}...")
    corpus_text = CORPUS_PATH.read_text()
    state.sources[CORPUS_PATH.name] = corpus_text
    chunks = state.splitter.split_text(corpus_text)
    metas  = [{"source": CORPUS_PATH.name, "chunk": i} for i, _ in enumerate(chunks)]
    store  = TurboQuantVectorStore.from_texts(chunks, state.embeddings, metadatas=metas)
    save_source(CORPUS_PATH.name, corpus_text)
    print(f"{len(chunks)} chunks indexed.")
    return store


# ---------------------------------------------------------------------------
# Mutable application state — a single namespace imported by all modules.
# Starts empty; populated by initialize() called from FastAPI lifespan.
# ---------------------------------------------------------------------------
state = types.SimpleNamespace()
state.ready = False


def initialize() -> None:
    """Load models and index. Runs in a thread after uvicorn binds port 8000."""
    _chunk_size, _chunk_overlap, _contextual = load_settings()
    state.chunk_size    = _chunk_size
    state.chunk_overlap = _chunk_overlap
    state.contextual    = _contextual
    state.splitter      = RecursiveCharacterTextSplitter(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
    state.sources       = load_sources()

    print("Loading embedding model...")
    state.embeddings = LocalEmbeddings()
    print("Loading re-ranker...")
    state.reranker = CrossEncoder(RERANKER_MODEL)

    if INDEX_PATH.exists():
        print(f"Loading saved index from {INDEX_PATH}...")
        state.store = TurboQuantVectorStore.load(INDEX_PATH, state.embeddings)
        # Detect dim mismatch — happens when embedding model changes.
        _embed_dim = len(state.embeddings.embed_query("test"))
        if state.store._index.dim is not None and state.store._index.dim != _embed_dim:
            print(
                f"Embedding model changed ({state.store._index.dim}-dim → {_embed_dim}-dim). "
                "Clearing incompatible index and re-indexing corpus..."
            )
            shutil.rmtree(INDEX_PATH)
            state.store = _build_index_from_corpus()
        else:
            print(f"{len(state.store._docs)} documents loaded.")
            if CORPUS_PATH.name not in state.sources and CORPUS_PATH.exists():
                corpus_text = CORPUS_PATH.read_text()
                state.sources[CORPUS_PATH.name] = corpus_text
                save_source(CORPUS_PATH.name, corpus_text)
    else:
        state.store = _build_index_from_corpus()

    for src_name, src_text in state.sources.items():
        if src_name.lower().endswith(".pdf"):
            ocr_path = OCR_DIR / (src_name + ".txt")
            if not ocr_path.exists():
                ocr_path.write_text(src_text)
                print(f"Backfilled OCR cache: {ocr_path.name}")

    state.llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)
    state.ready = True
    print("Initialization complete.")
