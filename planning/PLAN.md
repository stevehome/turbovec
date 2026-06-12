# Plan: Simple RAG Web Interface

## Goal

A local web app that loads a text corpus, stores it in a turbovec index, and lets the user query it with natural language. Claude answers using retrieved context.

## Stack

- **Gradio** — single-file Python UI, no separate frontend needed
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings, no API cost
- **turbovec** (`TurboQuantVectorStore`) — vector store
- **langchain-anthropic** (`ChatAnthropic`) — LLM for answers

## Phase 1: corpus from file, query via UI

### What it does

1. Loads a plain text file — one document per line (up to ~40 lines).
2. Embeds and indexes all lines into a `TurboQuantVectorStore` on startup.
3. Shows a text input. User types a question, hits Submit.
4. Retrieves top-k most relevant lines (k=3).
5. Passes retrieved context + question to Claude, displays the answer and the source lines used.

### UI layout

```
┌─────────────────────────────────────────────┐
│  Corpus: data/corpus.txt  (40 lines loaded) │
├─────────────────────────────────────────────┤
│  Question: [___________________________] [Ask] │
├─────────────────────────────────────────────┤
│  Answer:                                    │
│  ...Claude's response...                    │
│                                             │
│  Sources:                                   │
│  · line retrieved 1                         │
│  · line retrieved 2                         │
│  · line retrieved 3                         │
└─────────────────────────────────────────────┘
```

### Files

```
turbovec-python/
  app/
    app.py          # Gradio app — entry point
    data/
      corpus.txt    # default corpus, one line per document
```

### Run

```bash
uv pip install gradio sentence-transformers langchain-anthropic "langchain-core>=0.3"
uv run python app/app.py
```

## Phase 2: add and upload text (later)

Extend the UI with two extra inputs below the query box:

- **Add text** — a multi-line text area + "Add to index" button. Each non-empty line is embedded and added to the live index. No restart required.
- **Upload file** — a file upload widget accepting `.txt`. Lines are extracted and added to the index the same way as manual text.

The index is held in memory for the session; a "Save index" button calls `store.dump(path)` to persist it.

## Notes

- One `TurboQuantVectorStore` instance shared across the Gradio session (module-level).
- The embedding model is loaded once at startup (`scope="module"` equivalent — just a module-level variable).
- `bit_width=4` default; can be exposed as an advanced setting later.
- Keep `k=3` for retrieval — enough context without padding the prompt.
