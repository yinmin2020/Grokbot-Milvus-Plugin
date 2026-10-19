---
name: milvus-ops-inspect
description: Milvus 运维只读巡检：集群版本和健康检查、Collection 列表（行数/加载状态/索引）、Schema 详情、分区、Segment、Compaction 状态、/metrics 监控指标汇总、用户与角色。所有命令都是只读的，可以随时执行。当用户问 Milvus 是否健康、有哪些表、表结构、索引状态、监控指标、谁有什么权限时使用。
---

# Milvus 运维巡检（只读）

脚本：`scripts/milvus_ops.py`（`uv run` 执行）。连接参数用 `MILVUS_URI` / `MILVUS_TOKEN` / `MILVUS_DB`。
监控指标地址用 `MILVUS_METRICS_URL`，默认是 `http://<host>:9091/metrics`；端口做过映射时要显式设置，例如 `http://your-milvus-host:9092/metrics`。

| 命令 | 输出 |
| --- | --- |
| `uv run scripts/milvus_ops.py health` | 服务端版本、构建信息、部署模式、RPC 延迟、数据库列表、Collection 数量、`/healthz` 状态 |
| `uv run scripts/milvus_ops.py collections [--json]` | 每个 Collection 的行数（已加载时用 `count(*)`，否则用 flush 后的统计值）、加载状态、索引（字段:类型/度量），并标记受保护的 Collection |
| `uv run scripts/milvus_ops.py describe <col>` | 字段、Function（BM25/Embedding）、属性、分区、索引及构建进度、别名、一致性级别 |
| `uv run scripts/milvus_ops.py partitions <col>` | 分区及行数 |
| `uv run scripts/milvus_ops.py segments <col>` | 持久化 Segment（行数、状态、level、是否已排序）和已加载 Segment |
| `uv run scripts/milvus_ops.py compaction-state <job_id>` | Compaction 任务状态 |
| `uv run scripts/milvus_ops.py metrics [--grep REGEX]` | 汇总：节点数、Collection/实体/分区数、Segment 状态、存储量、Compaction 任务、请求成功和失败数、检索/写入平均延迟、内存、goroutine。加 `--grep` 可以查看原始指标 |
| `uv run scripts/milvus_ops.py users` | 用户及其角色、角色权限（不会显示密码） |

也可以用 MCP 工具：`milvus_list_collections`、`milvus_get_collection_info`、`milvus_list_databases`。

## 解读要点

- 刚写入、还没 flush 的数据在 `get_collection_stats` 里可能显示为 0，属于正常现象。以 `collections` 里的 `count(*)` 为准（需要 Collection 已加载）。
- 索引 `pending_index_rows > 0` 表示索引还在构建中。
- `milvus_rootcoord_force_deny_writing_counter > 0` 或磁盘配额告警，说明写入被限流了，要检查磁盘和内存。
- 指标端口连不上时，说明原因（端口映射、防火墙、代理）。不要编造数字。脚本访问内网地址时会自动绕过系统代理。
- 巡检报告格式：先给结论（健康 / 有风险），再列关键数字，最后给建议。
