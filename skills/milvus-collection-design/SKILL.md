---
name: milvus-collection-design
description: Design and create explicit Milvus collections for vectors, native BM25, or hybrid retrieval.
---

# milvus-collection-design

## Execution and configuration

Resolve `PLUGIN_ROOT` to the plugin directory containing `pyproject.toml`, two levels above this SKILL.md. Use an absolute path. The user's current directory may be a different project. Run scripts as:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/SCRIPT.py" --help
```

Replace SCRIPT with the entry point below. Connection settings are process environment variables: `MILVUS_URI` (default `./milvus.db`), `MILVUS_TOKEN`, and `MILVUS_DB` (default `default`). Use an absolute Lite path for repeatable connections from different working directories. The database lives on the machine executing the script. Every script closes its client before exiting; serialize commands sharing a Lite path.

Respect `OPS_PROTECTED_COLLECTIONS` and `OPS_WRITE_ALLOW_PREFIX` for every write. Confirm the user's intended collection and action before destructive changes; existing explicit authorization is sufficient. Do not print tokens or embed them in commands. Inspect `--help` for argument details. Do not invent commands or fall back to generated database code.

## Workflow

Identify the data, embedding dimension, primary key, metadata filters, and retrieval mode. Inspect existing collections with `milvus_ops.py describe COLLECTION` before proposing reuse. The data toolkit creates a fixed, explicit schema with `pk` (VARCHAR), `text` (VARCHAR), and `metadata` (JSON). Dense mode adds `dense`; BM25 mode adds a native analyzer/function and `sparse`; hybrid adds both. Dense indexes use AUTOINDEX/COSINE. The multilingual analyzer uses jieba.

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_data.py" create --collection articles --mode hybrid --dim 1536
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_data.py" create --collection articles --mode hybrid --dim 1536 --execute
```

Use the actual embedding model dimension; 1536 is only an example. For keyword-only retrieval use `--mode bm25` without a dimension or embedding model. Creation refuses existing names. For documents use the dedicated knowledge-base schema; for personal memory use the dedicated memory schema. Arbitrary schemas and server-side embedding functions are outside this creator's scope. Explain a proposed custom schema without claiming the preset creator can deploy it.
