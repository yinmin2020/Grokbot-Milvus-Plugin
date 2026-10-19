---
name: milvus-knowledge-desk
description: 知识库管家（Knowledge Desk）工作流。把 md/txt/pdf/docx 文档解析、分块，用 Ollama 或 OpenAI 兼容接口生成 embedding，写入 Milvus 混合检索 Collection（稠密向量 + BM25），按文档哈希增量更新；回答问题时做混合检索，并给出带引用的 chunk。当用户要导入文档、建知识库、基于文档问答、查看或删除知识库文档时使用。
---

# 知识库管家（Knowledge Desk）

脚本：`scripts/milvus_kb.py`（用 `uv run` 执行，会自动安装 pymilvus、pypdf、python-docx）。

## 0. 配置（环境变量，或对应的命令行参数）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MILVUS_URI` / `MILVUS_TOKEN` / `MILVUS_DB` | `http://localhost:19530` / 空 / `default` | 跟插件变量一致。token 是密钥，不要打印 |
| `EMBED_PROVIDER` | `ollama` | `ollama` 调 `/api/embed`；`openai` 调任意 OpenAI 兼容服务的 `/embeddings`（OpenAI、vLLM、Xinference、Ollama 的 `/v1` 等） |
| `EMBED_ENDPOINT` | `http://localhost:11434`（ollama）/ `https://api.openai.com/v1`（openai） | |
| `EMBED_MODEL` | 无 | 例如 `embeddinggemma-2:740m`（768 维）。新建 Collection 时必须提供；之后会从 Collection 描述里自动读取 |
| `EMBED_API_KEY` | 空 | 只有 openai provider 需要（也可以用 `OPENAI_API_KEY`），不要打印 |
| `KB_COLLECTION` | `kb_docs` | 默认知识库 Collection，也可以每次用 `--collection` 指定 |
| `KB_CHUNK_SIZE` / `KB_CHUNK_OVERLAP` | 500 / 80（字符） | 只影响新建的 Collection；已有 Collection 沿用建表时的设置 |
| `KB_MIN_SCORE` | 读 Collection 描述（embeddinggemma 为 0.62，其他模型为 0.5） | 最高 dense cosine 低于这个值时，判定为「没找到」 |

## 1. 导入（ingest）

```bash
uv run scripts/milvus_kb.py ingest docs/ --dry-run      # 只解析和分块，不写入，先给用户看
uv run scripts/milvus_kb.py ingest docs/ handbook.pdf   # 真正写入；Collection 不存在时自动创建
```

- **解析**：Markdown 按标题层级切分，保留标题路径；docx 识别 Heading/标题样式和表格；PDF 逐页提取，并记录页码；txt 按段落切分，首行较短时当作标题。扫描版 PDF 提取不到文字，会报 `[empty]`，需要先 OCR。
- **分块**：不会跨越标题章节或 PDF 页；长段落按句子切分；相邻块之间有重叠（overlap），重叠部分从句子边界开始。每个块的 `text` 开头带 `[标题路径]`，这样 BM25 和向量检索都能用到标题信息。
- **Schema**（Collection 由脚本创建）：`pk`（`doc_id#00001`）、`doc_id`（来源路径的 sha1）、`doc_hash`、`source`、`title`、`heading`、`page`、`chunk_no`、`lang`（zh/en）、`updated_at`、`text`（jieba + lowercase 分词器，启用 BM25）、`dense`（FLOAT_VECTOR，COSINE）、`sparse`（BM25 Function 自动生成）。字段名 `sparse` 和 `dense` 跟 MCP 工具 `milvus_text_search` / `milvus_hybrid_search` 兼容。
- **模型信息记录在 Collection 描述里**（JSON 格式：provider、model、dim、query/doc 前缀、分块参数、min_score）。之后再导入或查询时，如果用了不同的模型，脚本会直接拒绝，防止向量空间混用。
- **增量更新**：`doc_hash` 由文件内容和分块参数一起算出。内容没变就跳过（`[unchanged]`）；内容变了就按确定性的 pk 执行 upsert，再删除旧版本多出来的 chunk（`stale chunks deleted`）。加 `--force` 可以强制重新向量化。
- `--source-prefix https://wiki.example.com/` 可以把来源记录成 URL，方便引用时直接点开。

## 2. 检索与回答（query）

```bash
uv run scripts/milvus_kb.py query "HNSW 的 M 一般设多少？" --json            # 默认 hybrid + RRF，top-k 5
uv run scripts/milvus_kb.py query "..." --ranker weighted --weights 0.7,0.3  # 语义优先
uv run scripts/milvus_kb.py query "ERR_CODE_42" --mode bm25                  # 精确关键词
uv run scripts/milvus_kb.py query "..." --filter 'lang == "zh"'              # 也可以用 source like "docs/%"
```

返回字段：`found`、`best_dense_cosine`、`results[]`（每条包含 `cite`、`source`、`heading`、`page`、`text`、`score`、`dense_cosine`）。

回答规则：
1. 只依据 `results` 里的内容回答，每条结论后面标 `[cite]`，末尾列出 `[n] source > heading (p.页码)`。
2. `found=false` 或结果跟问题无关时，回答「知识库中没有找到相关内容」，并给出最高相似度。不要猜。
3. 跨语言提问（中文问、英文文档）一般也能检索到。如果没找到，可以用文档的语言换个说法再查一次。
4. 换了模型或文档类型后，用几个明显无关的问题（例如「如何做红烧肉」）校准 `--min-score`。
5. 不同文档说法不一致时，把各方说法连同出处并列出来，提示用户确认哪份是最新的。
6. 原文最多摘一两句，重点用自己的话归纳；不要用常识补充知识库里没有的事实，确需补充时单独标明「（以下不是知识库内容）」。
7. 入库前先说明要处理哪些文件、写入哪个 Collection；不要输出 `MILVUS_TOKEN`、`EMBED_API_KEY` 等密钥。

## 3. 维护

```bash
uv run scripts/milvus_kb.py info          # 模型、维度、行数、文档数、加载状态
uv run scripts/milvus_kb.py list-docs     # 每个文档的 chunk 数和 hash
uv run scripts/milvus_kb.py delete-doc --source docs/old.md                                      # dry-run，显示要删多少 chunk
uv run scripts/milvus_kb.py delete-doc --source docs/old.md --execute --confirm <collection>    # 用户确认后再执行
```

删除或重建整个 Collection 属于运维操作，交给 `milvus-ops-maintenance`（`drop-collection` 需要二次确认）。

## 4. Embedding 放在服务端还是客户端？

- 默认在**客户端**生成：脚本调用 Ollama 或 OpenAI 兼容接口生成向量，再写入 Milvus。任何 Milvus 版本（包括 Milvus Lite 和 Zilliz Cloud）都能用。
- Milvus 2.6+ 的 TEXTEMBEDDING Function 是在**服务端**生成向量。官方文档列出的 provider 有 OpenAI、Azure OpenAI、Bedrock、Vertex AI、Gemini、Voyage、Cohere、DashScope、SiliconFlow、Hugging Face TEI 等，**没有 Ollama**。另外两条路也走不通：`openai` provider 的自定义 `url` 和 API key 只能在服务端的 `milvus.yaml` 里配置；`TEI` provider 调用的是 TEI 的 `/embed` 接口，Ollama 不支持这个接口。而且服务端必须能访问到 embedding 服务。实测结论见 README。
