# Milvus plugin for Grok Bot & Cursor

Give the Grok Bot or Cursor agent a [Milvus](https://milvus.io) vector database. Point it at self-hosted Milvus,
**Zilliz Cloud**, or a local **Milvus Lite** file, and the agent can manage collections, insert data, and run
vector, full-text (BM25) and hybrid search. It can also build RAG knowledge bases and agent long-term memory.

The plugin bundles:

- **MCP server**: the open-source [zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus)
  (Apache-2.0). It's launched with `uvx` straight from GitHub at a pinned commit, so you don't need to clone it.
- **8 skills**: task guides that only reference tools and scripts that really exist:

| Skill | What it covers |
| --- | --- |
| `milvus-collection-design` | schema design, quick vs custom collections, indexes, BM25 collection recipe |
| `milvus-insert-data` | insert, delete, upsert-style updates, filter syntax |
| `milvus-search` | vector, BM25 full-text, hybrid (dense + BM25 sparse, RRF), filtered query |
| `milvus-rag-knowledge-base` | chunk → embed → ingest → retrieve → cited answers |
| `milvus-agent-memory` | long-term memory schema, remember / recall / update / forget |
| `milvus-knowledge-desk` | knowledge-base workflow: `scripts/milvus_kb.py` ingest / query / info / delete-doc (Chinese) |
| `milvus-ops-inspect` | read-only ops: health, collections, schema, segments, metrics, users (Chinese) |
| `milvus-ops-maintenance` | guarded ops changes: index, load/release, partitions, flush, compaction, delete, drop, backup (Chinese) |

- **Two script toolkits** (run with `uv run`, dependencies are declared inline):
  - **Knowledge Desk** (`scripts/milvus_kb.py`): ingests md/txt/pdf/docx into a hybrid (dense + BM25) collection
    with client-side embeddings from Ollama or any OpenAI-compatible endpoint, and returns cited chunks so the
    agent answers only from retrieved content.
  - **Ops** (`scripts/milvus_ops.py`): health, metrics, schema, indexes, partitions, compaction and more. Every
    write is a dry-run by default, and destructive commands need `--execute --confirm <collection>`.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH` (`uvx` comes with it), and `git`
  (uvx fetches the server from GitHub). The first launch downloads the dependencies (~80 packages); later
  launches use the cache.
- One of: Milvus Standalone/Cluster (2.5+, 2.6 recommended), Zilliz Cloud, or Milvus Lite (macOS/Linux only).

## Configuration (plugin variables)

Set these when you configure the plugin (in Cursor: **Dashboard → Plugins → Milvus → Configure**):

| Variable | Required | Example | Notes |
| --- | --- | --- | --- |
| `MILVUS_URI` | yes | `http://localhost:19530` | Milvus endpoint, Zilliz Cloud public endpoint, or absolute path to a Milvus Lite `.db` file |
| `MILVUS_TOKEN` | no (secret) | `root:Milvus` / Zilliz API key | Leave empty for local Milvus without auth and for Milvus Lite |
| `MILVUS_DB` | no | `default` | Database name |

Note: `MILVUS_URI` is passed as `--milvus-uri`, not as an environment variable, because pymilvus checks the
`MILVUS_URI` env var at import time and rejects Milvus Lite file paths.

### Milvus Standalone (Docker)

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

`MILVUS_URI=http://localhost:19530`, `MILVUS_TOKEN` empty. If you turned on authentication, use
`MILVUS_TOKEN=root:Milvus` (or `user:password`).

### Zilliz Cloud

1. Create a cluster at [cloud.zilliz.com](https://cloud.zilliz.com) (the Free tier works).
2. Copy the cluster's **Public Endpoint** (for example `https://in03-xxxx.serverless.<region>.cloud.zilliz.com`) and
   an **API key**, or a cluster `user:password`.
3. Set `MILVUS_URI=<public endpoint>` and `MILVUS_TOKEN=<API key>`.

An `https://` URI turns on TLS automatically in pymilvus, and the server passes the token straight to
`MilvusClient(uri=..., token=...)`.

### Milvus Lite (local file, macOS/Linux)

Set `MILVUS_URI` to an **absolute** path such as `/path/to/milvus/demo.db` (the folder must exist). The plugin
installs `milvus-lite` automatically on macOS and Linux. Milvus Lite has no Windows build, so on Windows use
Docker or Zilliz Cloud. Only one process can open a Lite file at a time.

## Tools exposed (14)

| Tool | Purpose |
| --- | --- |
| `milvus_list_databases` / `milvus_use_database` | list or switch databases |
| `milvus_list_collections` / `milvus_get_collection_info` | list collections, describe schema |
| `milvus_create_collection` | quick setup (`dimension`) or custom `field_schema` + `index_params` |
| `milvus_load_collection` / `milvus_release_collection` | load into or release from memory |
| `milvus_insert_data` | insert rows (list of objects) |
| `milvus_delete_entities` | delete by filter expression |
| `milvus_query` | scalar filter query |
| `milvus_vector_search` | dense vector search with filter and range options |
| `milvus_text_search` | BM25 full-text search (the sparse field must be named `sparse`) |
| `milvus_hybrid_search` | BM25 sparse + dense, RRF fusion, with filter |
| `milvus_text_similarity_search` | raw text query using a Milvus 2.6 server-side embedding function |

**Not available in the upstream server**: upsert, drop collection, create index (after creation), partitions,
creating BM25/embedding functions, and computing embeddings. The bundled scripts fill these gaps:
`milvus_kb.py` handles embeddings, BM25 collections and upsert, and `milvus_ops.py` handles indexes, partitions,
compaction and drop. The skills also give delete-then-insert and pymilvus fallbacks.

## Knowledge Desk: `scripts/milvus_kb.py`

```bash
export MILVUS_URI=http://localhost:19530 EMBED_PROVIDER=ollama EMBED_ENDPOINT=http://localhost:11434 EMBED_MODEL=embeddinggemma-2:740m
uv run scripts/milvus_kb.py ingest docs/ --dry-run          # parse + chunk only
uv run scripts/milvus_kb.py ingest docs/ manual.pdf         # creates the collection if needed, upserts by doc hash
uv run scripts/milvus_kb.py query "How do I tune HNSW?" --json
uv run scripts/milvus_kb.py info | list-docs | delete-doc --source docs/x.md [--execute --confirm <col>]
```

- Schema: `pk`, `doc_id`, `doc_hash`, `source`, `title`, `heading`, `page`, `chunk_no`, `lang`, `updated_at`,
  `text` (jieba + lowercase analyzer, so Chinese and English both work), `dense` (COSINE), and `sparse` (generated
  by a BM25 function). The field names match the MCP tools `milvus_text_search` and `milvus_hybrid_search`.
- The embedding provider, model, dim, task prefixes, chunking settings and "found" threshold are stored as JSON
  in the collection description. Ingest and query refuse a different model.
- Re-ingest is idempotent. Unchanged files are skipped. A changed file is upserted, and its stale chunks are deleted.
- `EMBED_PROVIDER=openai` with `EMBED_ENDPOINT=<base url>/v1` and `EMBED_API_KEY` works with OpenAI, vLLM, Xinference,
  Ollama's `/v1` endpoint, and similar services.

## Ops toolkit: `scripts/milvus_ops.py`

| Level | Commands | Runs when |
| --- | --- | --- |
| read | `health`, `collections`, `describe`, `partitions`, `segments`, `compaction-state`, `metrics`, `users`, `backup-guide` | always |
| write | `index-create`, `load`, `partition-create`, `flush`, `compact` | `--execute` |
| destructive | `release`, `index-drop`, `partition-drop`, `delete-entities`, `drop-collection` | `--execute --confirm <collection>` |

Protected collections (comma-separated in `OPS_PROTECTED_COLLECTIONS`) are always refused. `OPS_WRITE_ALLOW_PREFIX` limits writes to collections with the given prefixes. Executed writes
are logged to `~/.milvus-ops-audit.log`. Metrics are read from `MILVUS_METRICS_URL`, which defaults to
`http://<host>:9091/metrics`.

## Server-side embedding (TEXTEMBEDDING) with Ollama?

The Milvus 2.6/3.0 docs list the following embedding-function providers: OpenAI, Azure OpenAI, Bedrock, Vertex AI,
Gemini, Voyage, Cohere, DashScope, SiliconFlow, Hugging Face TEI, and Yandex. **Ollama is not on the list.**

- The `openai` provider accepts a custom `url`, but only in the server's `milvus.yaml`, together with a credential.
- The `TEI` provider takes an `endpoint` per function, but it calls TEI's `/embed` API. Ollama doesn't serve that API.

It also only works if the Milvus server can reach the embedding host. Tested against Milvus 3.0.1:

- `openai` provider with a custom url failed with `missing credentials config`.
- `TEI` provider pointed at the Ollama host failed with `no route to host`. Ollama listens on 127.0.0.1 only.

So the scripts use **client-side** embeddings, which work with any Milvus version, Milvus Lite, and Zilliz Cloud.

## Examples

- "List my Milvus collections and describe `docs`."
- "Create a collection `articles` with a 768-dim COSINE vector, a `title` VARCHAR and a `year` INT64."
- "Find the 5 chunks in `kb` closest to this vector where `lang == "en"`."
- "Do a hybrid search in `kb` for 'index tuning' using the `sparse` and `dense` fields."
- "Remember that I prefer answers in Chinese." (uses the `milvus-agent-memory` skill)
- "Ingest ./docs into the knowledge base and tell me how compaction works, with citations." (Knowledge Desk)
- "Is Milvus healthy? Show collections, load state and search latency." (Ops)

## Test it yourself

```bash
uv run scripts/smoke_test.py --uri http://localhost:19530        # or a Zilliz endpoint with MILVUS_TOKEN=...
uv run scripts/smoke_test.py --uri "$PWD/demo.db"                 # Milvus Lite
```

The script starts the server exactly as `mcp.json` declares it and lists the tools. It then creates
`grokbot_plugin_test`, inserts rows, searches, queries, and drops **only** that collection. If the collection
already exists, it refuses to run.

End-to-end test for both toolkits. It needs a real Milvus server and an embedding endpoint:

```bash
MILVUS_URI=http://localhost:19530 EMBED_MODEL=embeddinggemma-2:740m [MILVUS_METRICS_URL=http://localhost:9091/metrics] \
  uv run scripts/e2e_test.py
```

The test creates only `grokbot_kb_test_<ts>` and `grokbot_ops_test_<ts>`. Every collection that already existed is
treated as protected, and writes are limited to the test prefixes. At the end it drops the two test collections and
checks that every pre-existing collection still exists with an unchanged row count.

Validate the manifest, skills and script references with `python3 scripts/validate_plugin.py .`.

To try the plugin in Cursor before publishing, copy this folder to `~/.cursor/plugins/local/milvus` and run
**Developer: Reload Window**.

## Troubleshooting

- **Server shows as failed or has no tools**: the upstream server connects at startup and exits if Milvus is
  unreachable. Check `MILVUS_URI`, the network or VPN, and the token.
- **"collection ... is in state 'released'"**: call `milvus_load_collection` first.
- **`only drop_ratio_search=0 is supported`** (Milvus Lite): pass `drop_ratio: 0` to `milvus_text_search`.
- **`No module named 'mcp.server.fastmcp'`**: the upstream server needs the MCP Python SDK below 2.0. The
  plugin pins it with `--with "mcp<2"`. Keep that pin if you edit `mcp.json`.
- A `.env` file in the server's working directory overrides `MILVUS_TOKEN`/`MILVUS_DB` (upstream behaviour).

## Credits & license

- MCP server: [zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus) © Zilliz, Apache-2.0,
  pinned at commit `6a2bff9`. This repo does not vendor its code.
- This plugin (manifest, skills, docs, scripts): Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Milvus is an LF AI & Data Foundation project. This is a community plugin maintained by a Milvus community
  ambassador. It is not an official Zilliz product.

---

## 中文说明

这个插件让 Grok Bot 或 Cursor 的 Agent 能直接操作 **Milvus 向量数据库**，支持自建 Milvus、**Zilliz Cloud** 和本地 **Milvus Lite**。它可以管理 Collection、写入数据，执行向量检索、BM25 全文检索和混合检索（稠密向量 + BM25 稀疏向量，RRF 融合），也可以用来搭建 RAG 知识库和 Agent 长期记忆。

### 包含内容

- **MCP Server**：官方开源的 [zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus)（Apache-2.0）。通过 `uvx` 从 GitHub 的固定 commit 直接启动，不需要手动 clone。
- **8 个 Skill**：建表与 Schema 设计、数据写入/删除/更新、检索（向量/全文/混合/过滤）、RAG 知识库、Agent 长期记忆，以及新增的 `milvus-knowledge-desk`（知识库管家）、`milvus-ops-inspect`（运维巡检）和 `milvus-ops-maintenance`（安全变更）。
- **两套脚本工具**（用 `uv run` 执行，依赖写在脚本头部）：
  - **知识库管家（Knowledge Desk）**：`scripts/milvus_kb.py` 解析 md/txt/pdf/docx 并分块，chunk 会保留来源、标题路径和页码。Embedding 由 Ollama（`/api/embed`）或 OpenAI 兼容接口生成。写入时建混合检索 Collection（稠密向量 + BM25 稀疏向量），模型名和维度记在 Collection 描述里，按文档哈希做 upsert 增量更新。查询走混合检索，返回带引用的 chunk，配合 `milvus-knowledge-desk` skill：只根据检索结果回答，必须标出处，找不到就直说。
  - **运维助手（Ops）**：`scripts/milvus_ops.py` 覆盖健康检查和版本、Collection（行数、加载状态、索引）、Schema、建索引和删索引、load/release、分区、compaction、flush、Segment、`/metrics` 指标汇总、用户和角色、备份建议。写操作默认只 dry-run；危险操作必须同时加 `--execute` 和 `--confirm <collection 名>`；受保护的 Collection（`OPS_PROTECTED_COLLECTIONS`）一律拒绝。

### 前置条件

- 安装 [uv](https://docs.astral.sh/uv/)（自带 `uvx`）和 `git`。首次启动会下载依赖。
- 以下任选其一：Milvus Standalone/集群（2.5+，推荐 2.6）、Zilliz Cloud、Milvus Lite（仅 macOS/Linux）。

### 配置变量（Cursor 中：Dashboard → Plugins → Milvus → Configure）

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `MILVUS_URI` | 是 | 本地填 `http://localhost:19530`；Zilliz Cloud 填 Public Endpoint（`https://...`）；Milvus Lite 填 `.db` 文件的绝对路径 |
| `MILVUS_TOKEN` | 否（密钥） | Zilliz Cloud 填 API Key，开启鉴权的 Milvus 填 `用户名:密码`（如 `root:Milvus`），本地无鉴权或 Lite 留空 |
| `MILVUS_DB` | 否 | 数据库名，默认 `default` |

- **Milvus Standalone**：运行 `bash standalone_embed.sh start` 后，URI 填 `http://localhost:19530`。
- **Zilliz Cloud**：在控制台复制集群的 Public Endpoint 和 API Key，分别填入 `MILVUS_URI` 和 `MILVUS_TOKEN`。使用 `https://` 地址时会自动启用 TLS。
- **Milvus Lite**：`MILVUS_URI` 填绝对路径，例如 `/path/to/milvus/demo.db`。Windows 不支持 Milvus Lite。

### 已知限制

上游 MCP Server **没有** upsert、删除 Collection、建表后再建索引、分区管理、创建 BM25/Embedding Function、生成向量这些工具。现在由插件自带的 `milvus_kb.py` 和 `milvus_ops.py` 补上；Skill 里也保留了「先删除再插入」和 pymilvus 的替代写法。

**服务端 Embedding Function 能不能直接用 Ollama？** 目前不能。按官方文档（2.6/3.0），TEXTEMBEDDING 支持的 provider 里没有 Ollama。`openai` provider 的自定义 url 只能在服务端的 `milvus.yaml` 里配置，而且必须配好凭证；`TEI` provider 调用的 `/embed` 接口 Ollama 不支持。在 Milvus 3.0.1 上实测：openai 方式报 `missing credentials config`，TEI 方式指向 Ollama 所在主机时报 `no route to host`（Ollama 默认只监听 127.0.0.1）。所以脚本统一在客户端生成 embedding，这种方式适用于任何 Milvus 版本、Milvus Lite 和 Zilliz Cloud。
BM25 全文检索要求稀疏字段名为 `sparse`；在 Milvus Lite 上需要设置 `drop_ratio: 0`。Milvus 连不上时，Server 启动就会退出。

### 本地测试

```bash
uv run scripts/smoke_test.py --uri http://localhost:19530
```

脚本只会创建并删除 `grokbot_plugin_test` 这一个测试 Collection，不会动其他数据。

两套脚本的端到端测试（需要真实的 Milvus 和 embedding 服务）：

```bash
MILVUS_URI=http://localhost:19530 EMBED_MODEL=embeddinggemma-2:740m MILVUS_METRICS_URL=http://localhost:9091/metrics uv run scripts/e2e_test.py
```

测试只会创建 `grokbot_kb_test_<时间戳>` 和 `grokbot_ops_test_<时间戳>` 两个 Collection。运行前已经存在的 Collection 都会被当作受保护对象，写操作只允许用在这两个测试前缀上。测试结束后删除这两个测试 Collection，并核对原有 Collection 都还在、行数没有变化。

### 致谢与许可

MCP Server 来自 [zilliztech/mcp-server-milvus](https://github.com/zilliztech/mcp-server-milvus)（© Zilliz，Apache-2.0）。本插件同样以 Apache-2.0 开源。本插件是 Milvus 社区布道师维护的社区插件，不是 Zilliz 官方产品。
