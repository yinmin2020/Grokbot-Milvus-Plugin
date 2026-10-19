---
name: milvus-ops-maintenance
description: Milvus 安全运维变更：建索引、删索引、加载和释放（load/release）、分区创建和删除、flush、compaction、按条件删除数据、删除 Collection，以及备份建议。所有写操作默认只演练（dry-run），危险操作必须加 --execute 和 --confirm <collection 名>，受保护的 Collection 会被直接拒绝。当用户要修改索引、释放内存、清理数据、压缩、备份时使用。
---

# Milvus 安全运维变更

脚本：`scripts/milvus_ops.py`。上游 MCP Server 没有这些工具（upsert、drop/create index、分区、compaction、flush、drop collection 等），所以统一用这个脚本处理。

## 三级安全机制

| 级别 | 命令 | 执行条件 |
| --- | --- | --- |
| 写（可恢复） | `index-create`、`load`、`partition-create`、`flush`、`compact` | 默认 dry-run；加 `--execute` 才真正执行 |
| 危险（不可恢复，或会影响线上检索） | `release`、`index-drop`、`partition-drop`、`delete-entities`、`drop-collection` | 默认 dry-run；必须同时加 `--execute` 和 `--confirm <完全一致的 collection 名>` |
| 受保护 | 环境变量 `OPS_PROTECTED_COLLECTIONS`（逗号分隔） | 所有写操作和危险操作都直接拒绝 |

- 设置 `OPS_WRITE_ALLOW_PREFIX=test_,tmp_` 后，只允许修改带这些前缀的 Collection（适合测试环境和共享集群）。
- 每次真正执行的写操作都会追加到 `OPS_AUDIT_LOG`（默认 `~/.milvus-ops-audit.log`）。日志里不含密钥。

## 标准流程（Agent 必须遵守）

1. 只读检查：运行 `describe <col>` 和 `collections`，向用户说明现状。
2. 不加 `--execute` 运行一次，把 dry-run 输出（影响范围、匹配行数）原样告诉用户。
3. 危险操作要等用户**明确说出 Collection 名并同意**，然后才加 `--execute --confirm <col>`。
4. 执行完再运行一次 `describe` 或 `collections`，确认结果。

## 常用命令

```bash
# 建索引（标量 INVERTED / 向量 HNSW）
uv run scripts/milvus_ops.py index-create kb_docs --field tag --index-type INVERTED --execute
uv run scripts/milvus_ops.py index-create kb_docs --field dense --index-type HNSW --metric COSINE --params '{"M":16,"efConstruction":200}' --execute
# 换索引：release → index-drop → index-create → load
uv run scripts/milvus_ops.py release kb_docs --execute --confirm kb_docs
uv run scripts/milvus_ops.py index-drop kb_docs --index dense --execute --confirm kb_docs
uv run scripts/milvus_ops.py load kb_docs --execute
# 分区
uv run scripts/milvus_ops.py partition-create kb_docs p2026 --execute
uv run scripts/milvus_ops.py partition-drop kb_docs p2026 --execute --confirm kb_docs
# 刷盘和压缩（删除大量数据后回收空间）
uv run scripts/milvus_ops.py flush kb_docs --execute
uv run scripts/milvus_ops.py compact kb_docs --execute --wait
# 按条件删除（dry-run 会显示匹配行数；空条件会被拒绝）
uv run scripts/milvus_ops.py delete-entities kb_docs --filter 'source like "old/%"'
# 删除 Collection（先备份）
uv run scripts/milvus_ops.py drop-collection tmp_x --execute --confirm tmp_x
```

## 备份建议

运行 `uv run scripts/milvus_ops.py backup-guide` 可以查看完整建议（只是建议，不会执行任何操作）。要点：

- 推荐使用官方的 [milvus-backup](https://github.com/zilliztech/milvus-backup) 工具：先 `check`，再 `create -n bk_日期 [-c col]`，恢复时用 `restore -s _restored` 恢复到带后缀的新 Collection，核对行数后再切换别名。
- 备份前先执行 `flush`。
- 知识库类的 Collection 也可以从源文档重新导入来恢复（源文件放在 git 或对象存储里）。
- 任何 drop 操作之前，都要先确认备份可以成功恢复。
