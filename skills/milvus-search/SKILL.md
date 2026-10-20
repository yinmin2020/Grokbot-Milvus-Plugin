---
name: milvus-search
description: Search Milvus with dense vectors, native BM25, hybrid fusion, and server-side metadata filters.
---

# milvus-search

## Execution and configuration

Resolve `PLUGIN_ROOT` to the plugin directory containing `pyproject.toml`, two levels above this SKILL.md. Use an absolute path. The user's current directory may be a different project. Run scripts as:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/SCRIPT.py" --help
```

Replace SCRIPT with the entry point below. Connection settings are process environment variables: `MILVUS_URI` (default `./milvus.db`), `MILVUS_TOKEN`, and `MILVUS_DB` (default `default`). Use an absolute Lite path for repeatable connections from different working directories. The database lives on the machine executing the script. Every script closes its client before exiting; serialize commands sharing a Lite path.

Respect `OPS_PROTECTED_COLLECTIONS` and `OPS_WRITE_ALLOW_PREFIX` for every write. Confirm the user's intended collection and action before destructive changes; existing explicit authorization is sufficient. Do not print tokens or embed them in commands. Inspect `--help` for argument details. Do not invent commands or fall back to generated database code.

## Workflow

Inspect the collection schema, vector fields, and index metric with `milvus_ops.py describe COLLECTION`. Use `milvus_data.py query` for scalar queries and `milvus_data.py search` for retrieval. Pass filters to Milvus before top-k selection. Treat a raw `--filter` as an explicit expression, not as a template for concatenating unescaped input.

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_data.py" search --collection articles --mode bm25 --text "BM25" --filter 'metadata["lang"] == "en"'
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_data.py" search --collection articles --mode hybrid --text "BM25" --vector-file /absolute/path/query-vector.json --ranker rrf
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_data.py" query --collection articles --filter 'pk == "article-1"'
```

The vector file contains one JSON array of numbers produced by the same model as the stored vectors. Use `--dense-field`, `--sparse-field`, `--metric`, and `--output-fields` for compatible existing collections with other field names. Weighted fusion is available with `--ranker weighted --weights 0.7,0.3`. BM25 needs no query embedding. Scores from COSINE, BM25 and fusion have different scales; do not compare them with a shared threshold. L2 is lower-is-better; COSINE/IP are higher-is-better on supported versions.

For text questions over imported documents use `milvus_kb.py query`, which handles query embeddings and citations. For personal memory always use `milvus_memory.py recall` so the mandatory user scope cannot be omitted.
