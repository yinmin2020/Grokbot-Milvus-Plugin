---
name: milvus-inspect
description: Check the Milvus connection and view collections, fields, indexes and load state without modifying data.
---

# Milvus connection and collection inspection

Resolve `PLUGIN_ROOT` to the absolute plugin directory containing `pyproject.toml`, two levels above this SKILL.md. Use the connection configured in the shell environment: `MILVUS_URI`, `MILVUS_TOKEN`, and `MILVUS_DB`. Lite defaults to `./milvus.db`; prefer an absolute path so every command opens the intended database. A local path or `localhost` belongs to the machine executing the script.

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_inspect.py" health
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_inspect.py" collections
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_inspect.py" describe team_docs
```

`health` verifies an SDK request against the configured database. `collections` lists visible collection names. `describe` returns schema, indexes, load state and backend statistics for the selected collection. Inspection does not load, release, create or delete collections. The connection is closed after each command. Opening a new Lite path initializes a local database, so confirm that the configured path is the intended one.

Use the reported error to explain a connection or permission failure. Do not infer cluster-wide health from a successful connection check. Statistics can lag writes; use the knowledge-base skill's document listing and information commands for imported content. Do not print tokens. Import, update and search documents through the `milvus-knowledge-base` skill.
