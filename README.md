# Milvus plugin for Grok Bot and Cursor

Use Milvus from a skill-capable host through Python scripts and `MilvusClient`. The plugin provides collection/data tools, a document knowledge base, explicit user-scoped memory, and maintenance commands. It has no separate web UI and does not install automatic conversation-memory hooks.

| Capability | Script | Behavior |
| --- | --- | --- |
| Collections and structured data | `scripts/milvus_data.py` | Explicit dense/BM25/hybrid schemas, native upsert, scalar query, filtered search |
| Document knowledge base | `scripts/milvus_kb.py` | Markdown/text/PDF/DOCX parsing, embeddings, chunk updates, cited retrieval |
| Personal memory | `scripts/milvus_memory.py` | Explicit remember/recall/list/forget, mandatory user scope |
| Inspection and maintenance | `scripts/milvus_ops.py` | Health, schema, metrics, indexes, partitions and guarded destructive commands |

All eight skills call these entry points. The scripts share connection configuration, write restrictions and native retrieval helpers. Each invocation opens and closes its client. There is no background database connection owned by the plugin.

## Install and run

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and permission for the host to execute shell commands. Milvus Lite runs locally on supported macOS/Linux platforms. Windows users should connect to a Milvus server or Zilliz Cloud. The database path and `localhost` refer to the machine running the scripts; a hosted Grok Bot computer does not automatically access your laptop.

```bash
git clone https://github.com/zilliztech/Grokbot-Milvus-Plugin.git
cd Grokbot-Milvus-Plugin
uv sync --locked
export PLUGIN_ROOT="$PWD"
export MILVUS_URI="$PWD/milvus.db"
uv run scripts/validate_plugin.py
```

For a local Cursor plugin, copy the repository (excluding `.venv` and local databases) into `~/.cursor/plugins/local/milvus/` and reload the window. Use the host's plugin import mechanism for Grok Bot. Plugin loading and shell execution depend on host version and policy. The manifest and skills are provided here; the tests exercise script behavior, not the host UI.

Commands can run from any directory:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_ops.py" collections
```

Keep `PLUGIN_ROOT` and a Lite `MILVUS_URI` absolute. Skills resolve the plugin root from their own location. Connection and embedding settings must be present in the process environment used by the host's shell; manifest settings do not configure a separate shell process.

## Try keyword search without an embedding model

```bash
uv run scripts/milvus_data.py create --collection articles --mode bm25 --execute
cat > /tmp/milvus-articles.json <<'JSON'
[{"pk":"intro","text":"Milvus supports native BM25 and hybrid search.","metadata":{"lang":"en"}}]
JSON
uv run scripts/milvus_data.py upsert --collection articles --file /tmp/milvus-articles.json --execute
uv run scripts/milvus_data.py search --collection articles --mode bm25 --text "BM25" --filter 'metadata["lang"] == "en"'
```

`create` and `upsert` preview unless `--execute` is supplied. Dense/hybrid data collections require `--dim` on creation and a `dense` vector on every row. Query vectors are supplied as a JSON array through `--vector-file`. The generic data toolkit does not generate embeddings. It creates a fixed schema (`pk`, `text`, `metadata`, and the selected vector fields); arbitrary custom schemas and server-side embedding functions are outside that creator's scope. Search/query can inspect compatible existing collections using field/metric/output options.

## Document knowledge base

You provide an embedding service and model for document ingestion and dense/hybrid questions. No embedding model is downloaded by the plugin. OpenAI-compatible example (provide the key through your environment, without committing it):

```bash
export EMBED_PROVIDER=openai
export EMBED_MODEL=text-embedding-3-small
# Set OPENAI_API_KEY or EMBED_API_KEY in the environment.
uv run scripts/milvus_kb.py ingest samples/kb --collection team_docs --dry-run
uv run scripts/milvus_kb.py ingest samples/kb --collection team_docs
uv run scripts/milvus_kb.py query "What is hybrid search?" --collection team_docs --json
uv run scripts/milvus_kb.py query "BM25" --collection team_docs --mode bm25 --json
uv run scripts/milvus_kb.py list-docs --collection team_docs
```

For Ollama, set `EMBED_PROVIDER=ollama`, `EMBED_MODEL` to an already installed embedding model, and optionally `EMBED_ENDPOINT` (default `http://localhost:11434`). An OpenAI-compatible endpoint defaults to `https://api.openai.com/v1`. Collection configuration records provider, model, dimension, task prefixes, analyzer and chunk settings in a collection property; existing server collections with compatible description metadata can still be read. Changing provider/model or vector dimension requires a new collection. Serving a different model under the same model name cannot be detected automatically.

Markdown headings and PDF page numbers are retained for citations. Scanned PDFs need OCR first. Document IDs use resolved absolute paths by default, so the same `docs/readme.md` in two projects stays separate and changing the current directory does not duplicate a document. To move a project across machines while retaining IDs, use a stable namespace and source root on every import:

```bash
uv run scripts/milvus_kb.py ingest /work/project/docs --collection team_docs --namespace project-a --source-root /work/project
```

Reuse a namespace only for the same logical project. `--source-prefix` affects citation display, not identity. Re-import verifies the complete expected chunk set and hashes before skipping; a failed batch can be retried. Updates use native upsert, then remove surplus chunks. Imports into one collection must be serialized: multi-batch writes are not transactional, so readers may see partial updates until retry completes. Empty/unextractable documents fail with a message and retain existing content. Use `delete-doc --doc-id ID` to preview removal, then add `--execute --confirm COLLECTION` when intended. Removing a local file does not implicitly delete it from Milvus.

Search supports `--mode dense|bm25|hybrid`, `--filter`, `--top-k`, and RRF or weighted fusion. BM25 does not call the embedding service. `--min-score` applies only to dense cosine search and needs corpus/model-specific calibration. `found` means candidates were retrieved; BM25/fusion scores are not answer-confidence values. The skill must read the returned passages, cite supporting sources, and acknowledge when they do not answer the question.

## Explicit memory

```bash
uv run scripts/milvus_memory.py init --collection personal_memory --execute
uv run scripts/milvus_memory.py remember --collection personal_memory --user-id alice --memory-id editor --file /absolute/path/preference.txt --execute
uv run scripts/milvus_memory.py recall "preferred editor" --collection personal_memory --user-id alice
uv run scripts/milvus_memory.py list --collection personal_memory --user-id alice
uv run scripts/milvus_memory.py forget --collection personal_memory --user-id alice --memory-id editor
```

Every record command requires a user ID. IDs are combined with memory IDs for the primary key; updates use native upsert. The user ID is a native partition key with a scalar index. Dense/BM25/hybrid retrieval applies the user filter on the server before top-k selection. Forget requires `--execute --confirm personal_memory`. User IDs must come from trusted session context. This is application-level scoping, not access control against someone with the database token; multi-user applications need a trusted service and database permissions. Installing the skill alone does not persist every conversation automatically.

## Connection and maintenance

| Environment variable | Default / purpose |
| --- | --- |
| `MILVUS_URI` | `./milvus.db`; use an absolute path for Lite, `http://localhost:19530` for a server, or a Cloud endpoint |
| `MILVUS_TOKEN` | Empty; server `user:password` or Cloud API key |
| `MILVUS_DB` | `default` |
| `EMBED_PROVIDER`, `EMBED_MODEL` | Provider defaults to `ollama`; model must be supplied for new KB/memory collections |
| `EMBED_ENDPOINT` | Provider-specific URL |
| `EMBED_API_KEY` | Falls back to `OPENAI_API_KEY`; never stored in collection configuration |
| `EMBED_BATCH`, `EMBED_TIMEOUT` | `16`, `120` seconds |
| `KB_COLLECTION` | `kb_docs` |
| `OPS_PROTECTED_COLLECTIONS` | Comma-separated collection names that all entry points refuse to mutate |
| `OPS_WRITE_ALLOW_PREFIX` | Optional comma-separated write prefixes, shared by all entry points |
| `OPS_AUDIT_LOG` | `~/.milvus-ops-audit.log`; executed maintenance operations only |
| `MILVUS_METRICS_URL` | Server Prometheus endpoint, normally `http://host:9091/metrics` |

CLI connection flags override environment values. Reads used for retrieval and import verification request Strong consistency; this favors read-after-write correctness over latency. Ordinary writes do not flush. The explicit maintenance `flush` command is available when needed.

```bash
uv run scripts/milvus_ops.py health
uv run scripts/milvus_ops.py describe team_docs
uv run scripts/milvus_ops.py load team_docs
uv run scripts/milvus_ops.py load team_docs --execute
uv run scripts/milvus_ops.py drop-collection old_docs
```

All maintenance mutations preview by default. Destructive commands also require `--confirm COLLECTION`. Lite has no server metrics endpoint and does not implement every administrative API; Cloud availability also depends on the service and permissions. Run commands sharing a Lite path sequentially. These scripts close the connection on exit; another process holding the path can still cause a lock error. Use a Milvus server for concurrent workloads.

Existing Lite 2.x data is not assumed compatible with Lite 3.x: migrate or rebuild into a fresh path. If trying an earlier revision of this PR, use a new KB collection when re-ingesting documents: old working-directory-relative document IDs are not automatically migrated to the stable identity scheme. Existing records remain explicitly deletable by document ID.

## Validation

```bash
uv sync --locked
uv run scripts/validate_plugin.py
uv run ruff check scripts tests
uv run ruff format --check scripts tests
uv run scripts/smoke_test.py
uv run pytest -m 'not e2e'
# Real embeddings, fresh Lite database, isolated test collections:
EMBED_PROVIDER=openai EMBED_MODEL=text-embedding-3-small uv run scripts/e2e_test.py
# Optional server backend:
TEST_MILVUS_URI=http://localhost:19530 uv run pytest -m 'not e2e'
```

The tests create random collection names and remove only their own collections. Set `TEST_MILVUS_URI` and optional `TEST_MILVUS_TOKEN` to run against a server or Cloud. Real embedding E2E requires the configured service/key. Deterministic vectors are used only in regression tests for identity, interrupted writes, filtering and score semantics. No host UI test is implied by passing script tests.
