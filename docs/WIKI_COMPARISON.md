# turbovec-demo vs Karpathy LLM Wiki

Reference: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

## What each is

**turbovec-demo** is a classic retrieval-augmented generation (RAG) system:
- Raw text → chunk → embed → quantized vector index
- Each query: embed question → nearest-neighbour search → pass chunks to LLM → answer
- Knowledge lives as flat chunks; no synthesis between documents
- Fast and cheap to query; scales to large corpora via SIMD search

**Karpathy's LLM Wiki** is a compounding knowledge base:
- An LLM *reads* raw sources and *writes* structured markdown wiki pages
- Knowledge is synthesised, cross-referenced, and maintained by the LLM itself
- Queries may themselves become new wiki pages, so understanding compounds over time
- No vector search — just grep + index.md at small scale
- Follows an ingest → query → lint cycle; periodic lint passes catch contradictions, stale claims, and orphaned pages

## Key tradeoffs

| | turbovec-demo | LLM wiki |
|---|---|---|
| Query cost | Cheap (vector search + one LLM call) | Higher (reads wiki pages) |
| Knowledge synthesis | None — retrieves raw chunks | Yes — LLM summarises and cross-references |
| Scalability | Excellent (SIMD, millions of vectors) | Small-to-medium scale |
| Knowledge freshness | Re-chunk and re-index | Re-ingest and re-lint |
| Answers grounded in | Retrieved source chunks | LLM-maintained wiki pages |

## Bottom line

turbovec is better for *searching large corpora fast*; the LLM wiki is better for *building a living knowledge base* where synthesis and cross-referencing matter more than scale. They are complementary: turbovec could find relevant raw sources, while an LLM wiki maintains synthesised understanding on top of them.
