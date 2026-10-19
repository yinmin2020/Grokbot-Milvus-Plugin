---
name: milvus-rag-knowledge-base
description: Build and query a retrieval-augmented generation (RAG) knowledge base on Milvus - plan the collection, chunk and embed documents, ingest them, and answer questions with cited retrieved passages. Use when the user wants to index docs, a codebase, or notes into Milvus and ask questions over them.
---

# RAG knowledge base on Milvus

> **Ready-made pipeline:** for md/txt/pdf/docx files with an Ollama or OpenAI-compatible embedding model,
> prefer the `milvus-knowledge-desk` skill (`scripts/milvus_kb.py ingest|query`): hybrid dense + BM25
> collection, heading/page metadata, doc-hash upsert, cited results. Use the manual steps below for custom schemas.

Combines the `milvus-collection-design`, `milvus-insert-data` and `milvus-search` skills.

## Key constraint

The Milvus MCP server stores and searches vectors but **does not create embeddings**. Pick one strategy
up front and tell the user:

| Strategy | Ingest | Retrieve | Notes |
| --- | --- | --- | --- |
| A. Client-side embeddings (default) | write and run a small script (e.g. `sentence-transformers`, OpenAI, bge) to embed chunks, then insert via script or `milvus_insert_data` | embed the question with the same model, then `milvus_vector_search` | works everywhere, including Milvus Lite |
| B. BM25 only | create a BM25 collection (pymilvus snippet), insert raw text | `milvus_text_search` | no embedding model; keyword matching |
| C. Hybrid | BM25 collection + dense field | `milvus_hybrid_search` | best recall; needs both A and B |
| D. Server-side embedding | Milvus ≥ 2.6 with an embedding Function configured | `milvus_text_similarity_search` | needs server config by the admin |

## Instructions

1. **Plan** the collection (via `milvus-collection-design`). A good RAG schema:
   `pk` (VARCHAR, e.g. `"<doc_id>#<chunk_no>"`), `embedding` / `dense` (FLOAT_VECTOR, model dim),
   `text` (VARCHAR up to 8192–65535), `doc_id`, `source` (path or URL), `title`, `chunk_no` (INT64),
   `updated_at` (INT64 epoch seconds). Optional `sparse` + BM25 function for strategies B/C.
2. **Chunk** documents: ~300–800 tokens with 10–15% overlap; split on headings/paragraphs/code blocks;
   keep the title/heading path inside each chunk's text so it is self-describing.
3. **Embed & ingest**: for more than a few dozen chunks, write a script in the user's project that embeds
   and inserts with pymilvus (much faster than MCP tool calls). For small sets, `milvus_insert_data` in
   batches is fine. Use deterministic `pk`s so re-ingesting a document = `milvus_delete_entities` with
   `doc_id == "..."` followed by insert.
4. **Verify**: `milvus_query` with `doc_id == "..."` and `limit` to spot-check; run 2–3 known questions.
5. **Answer questions**:
   1. Retrieve top 5–10 chunks with the chosen strategy, using `filter_expr` for scoping (e.g.
      `source like "docs/%"`, `lang == "zh"`).
   2. Answer **only** from retrieved chunks; cite `source` / `pk` for each claim.
   3. If nothing relevant comes back, say so instead of guessing; suggest re-ingesting or widening filters.
6. **Maintenance**: deletes by `doc_id`; there is no drop-collection tool — if the user wants to rebuild
   from scratch, give them `MilvusClient(...).drop_collection("kb")` and confirm first.
