---
name: milvus-collection-design
description: Design and create Milvus collections (schema, primary key, vector fields, scalar fields, indexes) through the Milvus MCP server. Use when the user wants to create a collection, choose a vector dimension or metric, add metadata fields for filtering, or inspect an existing collection's schema.
---

# Milvus collection & schema design

## When to use

- "Create a collection for ..." / "design a schema for ..."
- Choosing dimension, metric type (COSINE / L2 / IP), primary key type, filterable metadata fields
- Inspecting what already exists before writing data

## Tools used (zilliztech/mcp-server-milvus)

`milvus_list_databases`, `milvus_use_database`, `milvus_list_collections`, `milvus_get_collection_info`,
`milvus_create_collection`, `milvus_load_collection`, `milvus_release_collection`.

There is **no** drop-collection, create-index, partition, or alias tool in this server. If the user needs
those, use the bundled guarded script `scripts/milvus_ops.py` (see the `milvus-ops-maintenance` skill: index-create,
index-drop, partition-create/-drop, release, drop-collection; dry-run by default) or give a short pymilvus snippet —
never pretend an MCP tool exists.

## Instructions

1. Always start with `milvus_list_collections` (and `milvus_get_collection_info` for any candidate) so you
   never try to create a name that already exists — creation fails if it does.
2. Ask (or infer) the embedding model, because **dimension must match the model** (e.g. 384 for
   all-MiniLM-L6-v2, 768 for many BERT models, 1024 for bge-m3 dense, 1536 for text-embedding-3-small).
   Default metric: `COSINE` for normalized text embeddings.
3. **Quick setup** (prototyping): call `milvus_create_collection` with only `collection_name`, `dimension`,
   optionally `metric_type`, `primary_field_name`, `vector_field_name`, `auto_id`.
   This gives an INT64 `id` primary key, a float vector field `vector`, an index, and a **dynamic field**,
   so any extra keys you insert (`text`, `source`, `tag`, ...) are stored and filterable. The collection is loaded and ready.
4. **Custom schema** (production): pass `field_schema` + `index_params`. Each `field_schema` item is passed
   straight to pymilvus `schema.add_field(**item)`, so use pymilvus argument names:
   - `field_name` (not `name`), `datatype` as a string DataType name (not `type`), e.g. `"INT64"`, `"VARCHAR"`,
     `"FLOAT_VECTOR"`, `"SPARSE_FLOAT_VECTOR"`, `"JSON"`, `"BOOL"`, `"FLOAT"`, `"ARRAY"`
   - `is_primary: true` on exactly one field; `max_length` for VARCHAR; `dim` for FLOAT_VECTOR
   - Optional: `nullable`, `default_value`, `is_partition_key`, `enable_analyzer`, `enable_match`
   `index_params` items: `{"field_name": "...", "index_type": "AUTOINDEX", "metric_type": "COSINE"}`
   (add `"params": {...}` for HNSW/IVF). Always include an index for every vector field, because you cannot
   add one later through MCP. Pass extra collection options (e.g. `{"enable_dynamic_field": false}`)
   via `other_kwargs`.
5. After creating a custom-schema collection, call `milvus_load_collection`, then
   `milvus_get_collection_info` to confirm the schema.
6. Tested example (custom schema):

```json
{
  "collection_name": "docs",
  "auto_id": false,
  "field_schema": [
    {"field_name": "pk", "datatype": "VARCHAR", "is_primary": true, "max_length": 64},
    {"field_name": "embedding", "datatype": "FLOAT_VECTOR", "dim": 768},
    {"field_name": "source", "datatype": "VARCHAR", "max_length": 512},
    {"field_name": "created_at", "datatype": "INT64"}
  ],
  "index_params": [
    {"field_name": "embedding", "index_type": "AUTOINDEX", "metric_type": "COSINE"}
  ]
}
```

## Full-text (BM25) collections — limitation

BM25 full-text search needs a `SPARSE_FLOAT_VECTOR` field produced by a BM25 **Function** on an analyzed
VARCHAR field. `milvus_create_collection` cannot declare functions, so **a BM25 collection cannot be
created through this MCP server**. Tell the user and offer this pymilvus snippet (run once; then all
search tools work on it). Name the sparse field `sparse`, because `milvus_text_search` hard-codes that name:

```python
from pymilvus import MilvusClient, DataType, Function, FunctionType
c = MilvusClient(uri="http://localhost:19530", token="")  # or Zilliz Cloud endpoint + API key
s = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
s.add_field("id", DataType.INT64, is_primary=True)
s.add_field("text", DataType.VARCHAR, max_length=8192, enable_analyzer=True)
s.add_field("dense", DataType.FLOAT_VECTOR, dim=768)
s.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
s.add_function(Function(name="bm25", function_type=FunctionType.BM25,
                        input_field_names=["text"], output_field_names=["sparse"]))
ip = c.prepare_index_params()
ip.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
ip.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
c.create_collection("kb", schema=s, index_params=ip)
```

## Guardrails

- Never call `milvus_release_collection` on a collection the user is serving traffic from without asking.
- Collection names: letters, digits, underscores; start with a letter or underscore.
