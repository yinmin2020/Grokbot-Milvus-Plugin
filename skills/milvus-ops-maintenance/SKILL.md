---
name: milvus-ops-maintenance
description: Plan and execute guarded Milvus index, partition, load, delete and compaction operations.
---

# milvus-ops-maintenance

## Execution and configuration

Resolve `PLUGIN_ROOT` to the plugin directory containing `pyproject.toml`, two levels above this SKILL.md. Use an absolute path. The user's current directory may be a different project. Run scripts as:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/SCRIPT.py" --help
```

Replace SCRIPT with the entry point below. Connection settings are process environment variables: `MILVUS_URI` (default `./milvus.db`), `MILVUS_TOKEN`, and `MILVUS_DB` (default `default`). Use an absolute Lite path for repeatable connections from different working directories. The database lives on the machine executing the script. Every script closes its client before exiting; serialize commands sharing a Lite path.

Respect `OPS_PROTECTED_COLLECTIONS` and `OPS_WRITE_ALLOW_PREFIX` for every write. Confirm the user's intended collection and action before destructive changes; existing explicit authorization is sufficient. Do not print tokens or embed them in commands. Inspect `--help` for argument details. Do not invent commands or fall back to generated database code.

## Guarded maintenance

Use `milvus_ops.py`. Every mutation defaults to a dry run. Index creation, load, partition creation, explicit flush and compaction need `--execute`. Release, index/partition deletion, entity deletion and collection deletion also need `--confirm COLLECTION`. The shared protected collection list and write prefix rules apply. Executed maintenance is appended to `OPS_AUDIT_LOG` (default `~/.milvus-ops-audit.log`); a failed log write emits a warning.

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" load team_docs
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" load team_docs --execute
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" index-create articles --field dense --index-type AUTOINDEX --metric COSINE
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" partition-create articles archive
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" delete-entities articles --filter 'pk == "obsolete"'
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" compact articles
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" drop-collection obsolete_docs
```

Inspect affected resources and show the dry-run plan before executing a destructive operation. Do not drop/recreate incompatible knowledge-base schemas without the user's authorization; prefer a new collection. Use `backup-guide` for guidance only; the script does not create or restore backups. Maintenance capabilities depend on the deployment. Do not automatically flush after ordinary writes, and do not convert a backend error into success.
