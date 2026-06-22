"""Global application state, path constants, and startup initialisation."""
from __future__ import annotations

import json
import types
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from turbovec.langchain import TurboQuantVectorStore

_HERE = Path(__file__).parent
CORPUS_PATH   = _HERE / "data" / "corpus.txt"
INDEX_PATH    = _HERE / "data" / "saved_index"
SETTINGS_PATH = _HERE / "data" / "settings.json"
SOURCES_PATH  = _HERE / "data" / "sources.json"
OCR_DIR       = _HERE / "data" / "ocr"
OCR_DIR.mkdir(exist_ok=True)
K = 3


class LocalEmbeddings(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()


def load_settings() -> tuple[int, int, bool]:
    if SETTINGS_PATH.exists():
        s = json.loads(SETTINGS_PATH.read_text())
        return int(s.get("chunk_size", 500)), int(s.get("chunk_overlap", 50)), bool(s.get("contextual", False))
    return 500, 50, False


def save_settings(chunk_size: int, chunk_overlap: int, contextual: bool) -> None:
    SETTINGS_PATH.write_text(json.dumps({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "contextual": contextual}))


def load_sources() -> dict[str, str]:
    if SOURCES_PATH.exists():
        return json.loads(SOURCES_PATH.read_text())
    return {}


def save_sources() -> None:
    SOURCES_PATH.write_text(json.dumps(state.sources))


def chunk_with_meta(text: str, source: str) -> tuple[list[str], list[dict]]:
    chunks = state.splitter.split_text(text)
    metas = [{"source": source, "chunk": i} for i, _ in enumerate(chunks)]
    return chunks, metas


# ---------------------------------------------------------------------------
# Mutable application state — a single namespace imported by all modules.
# ---------------------------------------------------------------------------
state = types.SimpleNamespace()

_chunk_size, _chunk_overlap, _contextual = load_settings()
state.chunk_size   = _chunk_size
state.chunk_overlap = _chunk_overlap
state.contextual   = _contextual
state.splitter     = RecursiveCharacterTextSplitter(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
state.sources      = load_sources()

print("Loading embedding model...")
state.embeddings = LocalEmbeddings()

if INDEX_PATH.exists():
    print(f"Loading saved index from {INDEX_PATH}...")
    state.store = TurboQuantVectorStore.load(INDEX_PATH, state.embeddings)
    print(f"{len(state.store._docs)} documents loaded.")
    if CORPUS_PATH.name not in state.sources and CORPUS_PATH.exists():
        state.sources[CORPUS_PATH.name] = CORPUS_PATH.read_text()
        save_sources()
else:
    print(f"Indexing corpus from {CORPUS_PATH}...")
    corpus_text = CORPUS_PATH.read_text()
    state.sources[CORPUS_PATH.name] = corpus_text
    chunks = state.splitter.split_text(corpus_text)
    metas  = [{"source": CORPUS_PATH.name, "chunk": i} for i, _ in enumerate(chunks)]
    state.store = TurboQuantVectorStore.from_texts(chunks, state.embeddings, metadatas=metas)
    save_sources()
    print(f"{len(chunks)} chunks indexed.")

for _src_name, _src_text in state.sources.items():
    if _src_name.lower().endswith(".pdf"):
        _ocr_path = OCR_DIR / (_src_name + ".txt")
        if not _ocr_path.exists():
            _ocr_path.write_text(_src_text)
            print(f"Backfilled OCR cache: {_ocr_path.name}")

state.llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)
