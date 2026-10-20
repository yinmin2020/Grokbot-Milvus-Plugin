---
name: milvus-ops-inspect
description: Inspect Milvus health, collections, schema, indexes, partitions, metrics and users.
---

# milvus-ops-inspect

## Execution and configuration

Resolve `PLUGIN_ROOT` to the plugin directory containing `pyproject.toml`, two levels above this SKILL.md. Use an absolute path. The user's current directory may be a different project. Run scripts as:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/SCRIPT.py" --help
```

Replace SCRIPT with the entry point below. Connection settings are process environment variables: `MILVUS_URI` (default `./milvus.db`), `MILVUS_TOKEN`, and `MILVUS_DB` (default `default`). Use an absolute Lite path for repeatable connections from different working directories. The database lives on the machine executing the script. Every script closes its client before exiting; serialize commands sharing a Lite path.

Respect `OPS_PROTECTED_COLLECTIONS` and `OPS_WRITE_ALLOW_PREFIX` for every write. Confirm the user's intended collection and action before destructive changes; existing explicit authorization is sufficient. Do not print tokens or embed them in commands. Inspect `--help` for argument details. Do not invent commands or fall back to generated database code.

## Read-only inspection

Use `milvus_ops.py` and explain results in terms of the user's requested diagnosis.

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" health
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" collections --json
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" describe team_docs
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" partitions team_docs
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" segments team_docs
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" metrics
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" users
```

For server metrics set `MILVUS_METRICS_URL` to the reachable Prometheus endpoint (usually port 9091). Lite has no server metrics endpoint and some administrative APIs are unavailable; managed services can also restrict them. Report unsupported operations as a deployment limitation, not as proof that data is unhealthy. Use `compaction-state JOB_ID` for an existing job. Do not automatically run maintenance as part of inspection.
