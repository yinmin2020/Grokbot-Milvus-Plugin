---
name: milvus-search
description: Run vector similarity, full-text (BM25), hybrid (dense + BM25 sparse) and filtered scalar searches in Milvus through the Milvus MCP server. Use when the user wants to search, retrieve similar items, do keyword or hybrid retrieval, or query records by metadata.
---

# Searching Milvus: vector, BM25, hybrid, filters

## Tools used

| Tool | Use for | Needs |
| --- | --- | --- |
| `milvus_query` | exact metadata lookups, no ranking | `filter_expr` |
| `milvus_vector_search` | dense similarity (+ optional filter, range search) | a query vector |
| `milvus_text_search` | BM25 full-text search | BM25 collection whose sparse field is named `sparse` |
| `milvus_hybrid_search` | BM25 + dense, fused with RRF (k=60) | BM25 sparse field + dense field + query vector |
| `milvus_text_similarity_search` | send raw text, Milvus embeds it | Milvus ≥ 2.6 with an embedding Function on the collection |

## Instructions

1. `milvus_get_collection_info` first: find vector field names, the `functions` list (BM25 / embedding
   function present?) and scalar fields usable in filters.
2. If search says the collection is not loaded (state "released"), call `milvus_load_collection` and retry.
   Collections are often released after a Milvus / Milvus Lite restart.
3. **Getting a query vector**: the MCP server has no embedding tool. Use, in order of preference:
   a. `milvus_text_similarity_search` if the collection has a server-side embedding function;
   b. a vector produced by the user's embedding code with the **same model used at insert time**;
   c. otherwise, `milvus_text_search` (BM25) if available, which needs no vector.
   Never fabricate a vector.
4. **Vector search**: `milvus_vector_search` with `collection_name`, `vector`, `vector_field` (default
   `"vector"`), `limit`, `output_fields`, optional `filter_expr`, and `metric_type` matching the index.
   Range search: set `radius` / `range_filter` (for COSINE, `radius` is the minimum similarity).
5. **Full-text (BM25)**: `milvus_text_search` with `query_text`. Limitations of the upstream tool:
   the sparse field name is hard-coded to `sparse`, and it has **no filter parameter**. On Milvus Lite set
   `drop_ratio: 0` (Lite rejects other values); on standalone/Zilliz Cloud the default 0.2 is fine.
6. **Hybrid (dense + BM25 sparse)**: `milvus_hybrid_search` with
   `query_text`, `text_field` = **the sparse BM25 output field (e.g. `sparse`), not the VARCHAR text
   field**, `vector`, `vector_field` (e.g. `dense`), `limit`, `output_fields`, optional `filter_expr`.
   Ranking is RRF with k=60 and cannot be changed (no weighted ranker exposed). Scores are RRF scores
   (~0.03), not similarities — do not threshold on them.
   If the collection has no BM25 function, hybrid search is not possible through this server; say so and
   point to the pymilvus snippet in the `milvus-collection-design` skill.
7. Always pass `output_fields` (e.g. `["text", "source"]`) so results are readable, and cite primary keys
   / sources when presenting results.

## Example (tested on Milvus Lite)

```json
{"collection_name": "kb", "query_text": "vector database", "text_field": "sparse",
 "vector": [0.1, 0.2, 0.3, 0.4], "vector_field": "dense", "limit": 5,
 "output_fields": ["text"], "filter_expr": "tag == \"db\""}
```
