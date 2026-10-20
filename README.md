# Milvus document knowledge base

Import your documents into Milvus, update them when files change, and retrieve passages with source citations from Grok Bot or Cursor. The host writes answers from the retrieved evidence. Markdown, text, PDF and DOCX are supported, with native dense, BM25 and hybrid retrieval.

The plugin has two skills and two command entry points:

| Skill | Entry point | Purpose |
| --- | --- | --- |
| `milvus-knowledge-base` | `scripts/milvus_kb.py` | Import, update, inspect and remove documents; retrieve cited passages |
| `milvus-inspect` | `scripts/milvus_inspect.py` | Check the connection, list collections and view collection details |

Both scripts call `MilvusClient` directly and close the connection on exit. Document parsing and chunking live in `kb_documents.py`; connection, embedding and validation helpers live in `mvbot_common.py`.

## Start on macOS or Linux

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), and a host permitted to execute shell commands. Milvus Lite runs locally on supported macOS/Linux platforms. A Milvus server or Zilliz Cloud can be selected with the same connection settings; Windows users should use a server endpoint.

```bash
git clone https://github.com/zilliztech/Grokbot-Milvus-Plugin.git
cd Grokbot-Milvus-Plugin
uv sync --locked
export PLUGIN_ROOT="$PWD"
export MILVUS_URI="$PWD/milvus.db"
uv run scripts/milvus_inspect.py health
```

Provide an embedding service for document ingestion and dense/hybrid retrieval. For an OpenAI-compatible service:

```bash
export EMBED_PROVIDER=openai
export EMBED_MODEL=text-embedding-3-small
# Supply OPENAI_API_KEY or EMBED_API_KEY through your environment.
uv run scripts/milvus_kb.py ingest samples/kb --collection team_docs --dry-run
uv run scripts/milvus_kb.py ingest samples/kb --collection team_docs
uv run scripts/milvus_kb.py query "What is hybrid search?" --collection team_docs --json
```

For Ollama, set `EMBED_PROVIDER=ollama` and `EMBED_MODEL` to an already installed embedding model; the default endpoint is `http://localhost:11434`. The plugin does not download models. OpenAI-compatible services default to `https://api.openai.com/v1`; use `EMBED_ENDPOINT` for another compatible service. The embedding dimension is detected from the model.

To install locally in Cursor, copy the repository, excluding `.venv` and local databases, into `~/.cursor/plugins/local/milvus/` and reload the window. Use Grok Bot's plugin import mechanism for that host. Loading and shell execution depend on host version and policy; script tests do not validate the host UI.

The connection and embedding variables must reach the host's shell process. A hosted computer's `localhost` does not refer to your laptop. From another working directory, run:

```bash
uv run --project "$PLUGIN_ROOT" "$PLUGIN_ROOT/scripts/milvus_kb.py" query "What is hybrid search?" --collection team_docs --json
```

Keep the plugin root and Lite database path absolute.

## Import, update and maintain documents

`ingest` creates the knowledge-base collection if needed. `--dry-run` parses and chunks without creating a collection, writing records or calling the embedding service. Scanned PDFs need OCR before import.

Run the same import after editing your files. Stable document IDs and native upsert preserve identity; complete unchanged documents are skipped and surplus chunks removed. By default, document identity uses the resolved absolute path, so identical relative names in different projects stay separate and changing the current directory does not duplicate files.

For a project that moves between machines, supply both a stable namespace and a source root on every import:

```bash
uv run scripts/milvus_kb.py ingest /work/project/docs --collection team_docs --namespace project-a --source-root /work/project
uv run scripts/milvus_kb.py info --collection team_docs
uv run scripts/milvus_kb.py list-docs --collection team_docs
```

Reuse a namespace only for the same logical project. `--source-prefix` changes the displayed citation source without changing document identity. Model, dimension, task prefixes and chunk settings are recorded on the collection. Incompatible schemas/models are rejected; create a new collection to change the embedding model. A provider serving different weights under the same model name cannot be detected automatically.

Imports into one collection must be serialized. Multi-batch writes are retryable but not transactional; readers can see partial updates until retry completes. Re-import verifies the complete expected chunk set before skipping. An empty/unextractable document reports an error and retains existing content. Removing a local file does not remove its stored chunks. To remove a document deliberately, use its ID from `list-docs`:

```bash
uv run scripts/milvus_kb.py delete-doc --collection team_docs --doc-id DOCUMENT_ID
uv run scripts/milvus_kb.py delete-doc --collection team_docs --doc-id DOCUMENT_ID --execute --confirm team_docs
```

The first command previews the affected chunk count. The second requires the exact collection name as confirmation.

## Retrieve and cite

```bash
uv run scripts/milvus_kb.py query "How does hybrid search work?" --collection team_docs --json
uv run scripts/milvus_kb.py query "BM25" --collection team_docs --mode bm25 --json
uv run scripts/milvus_kb.py query "向量检索" --collection team_docs --filter 'lang == "zh"' --json
```

Default hybrid retrieval combines dense embeddings and native BM25 with reciprocal rank fusion. Pure BM25 questions do not call the embedding service. Filters run in Milvus before top-k selection. Optional `--top-k`, `--candidates`, and `--ranker weighted --weights 0.7,0.3` tune retrieval. `--mode dense --min-score 0.5` applies a cosine threshold; calibrate it for your model and corpus.

Results contain a citation number, source, title, heading, PDF page, document ID, chunk number and the supporting text. Markdown/DOCX headings and PDF pages are preserved. `found` indicates retrieved candidates, not proof that they answer a question. The skill reads the passages, cites supported claims as `[1]`, `[2]`, and acknowledges insufficient evidence. BM25 and fusion scores are not universal confidence values. Retrieved documents are treated as data, including any instructions embedded in them.

## Connection and collection viewing

```bash
uv run scripts/milvus_inspect.py health
uv run scripts/milvus_inspect.py collections
uv run scripts/milvus_inspect.py describe team_docs
```

These commands verify connectivity and show collection names, schema, indexes, load state and backend statistics. They do not load or modify collections. Backend row statistics may lag recent writes. Opening a new Lite path initializes the database at that location.

| Environment variable | Default / purpose |
| --- | --- |
| `MILVUS_URI` | `./milvus.db`; prefer an absolute Lite path, or use a server/Cloud endpoint |
| `MILVUS_TOKEN` | Empty; server `user:password` or Cloud API key |
| `MILVUS_DB` | `default` |
| `EMBED_PROVIDER`, `EMBED_MODEL` | Provider defaults to `ollama`; specify the model for a new knowledge base |
| `EMBED_ENDPOINT` | Provider-specific URL |
| `EMBED_API_KEY` | Falls back to `OPENAI_API_KEY`; not stored in collection metadata |
| `EMBED_BATCH`, `EMBED_TIMEOUT` | `16`, `120` seconds |
| `KB_COLLECTION` | `kb_docs` |
| `KB_PROTECTED_COLLECTIONS` | Comma-separated collection names that knowledge-base writes must not change |
| `KB_WRITE_ALLOW_PREFIX` | Optional comma-separated allowed prefixes for knowledge-base writes |

CLI flags override environment settings. Import verification and retrieval request Strong consistency for read-after-write correctness; ordinary writes do not flush. Commands using the same Lite path must run sequentially. Use a server for concurrent workloads.

For a server, set `MILVUS_URI=http://localhost:19530`; for Cloud, use its endpoint and token. Lite 2.x data is not assumed compatible with Lite 3.x: migrate or rebuild into a fresh path. If you used an early revision with working-directory-relative document IDs, re-ingest into a new collection or explicitly clean up the old document IDs.

## Validation

```bash
uv sync --locked
uv run scripts/validate_plugin.py
uv run ruff check scripts tests
uv run ruff format --check scripts tests
uv run pytest -m 'not e2e'
# Real embedding service, fresh Lite database:
RUN_EMBEDDING_E2E=1 EMBED_PROVIDER=openai EMBED_MODEL=text-embedding-3-small uv run pytest
# Optional server backend, using the same suite:
TEST_MILVUS_URI=http://localhost:19530 uv run pytest -m 'not e2e'
```

The tests create unique collection names and remove only their own collections. `TEST_MILVUS_TOKEN` supplies a token for server/Cloud tests. Deterministic vectors cover regression cases such as interrupted imports, identity, filtering and citations. The opt-in E2E uses a real embedding service for all four document formats, retrieval, updates and deletion. CLI tests cover Lite close/reopen and execution outside the repository.
