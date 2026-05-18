# 建库进度与失败恢复

GraphRAG 建库是多 workflow 流程：文本切片、实体关系抽取、描述总结、图谱收尾、社区发现、社区摘要、向量索引等步骤会依次写出中间表。理解这些中间表，可以在失败后更准确地判断重跑范围。

## 1. 观察进度

建议建库时开启详细日志：

```bash
graphrag index --root . --verbose
```

本仓库额外增强了两处可观测性：

- 实体/关系抽取完成后，会记录已处理 chunk 数、合并后的实体数和关系数。
- 向量索引每次批量写入后，会通过 workflow callbacks 上报已完成 embedding 行数。

这些信息可以帮助判断卡在 LLM 抽取、社区摘要，还是 embedding 写入阶段。

## 2. 主要输出表

常见输出包括：

- `text_units`：文本切片后的原文片段。
- `entities`：归并、总结后的实体表。
- `relationships`：归并、加权后的关系表。
- `communities`：社区发现结果。
- `community_reports`：社区摘要。
- `graph.graphml`：可视化图文件。
- `embeddings.*`：如果开启 `snapshots.embeddings`，会保存 embedding 快照。

如果某一步失败，先看失败点之前的表是否已经正常写出，再决定是否只重跑后续 workflow。

## 3. 恢复建议

GraphRAG 的 LLM 调用会通过 cache 复用已有结果。失败后重跑同一输入和同一配置时，已缓存的 completion/embedding 通常不会重新消耗完整调用成本。

恢复时建议遵循以下顺序：

1. 不要先删除 `cache`、`output`、`logs`。
2. 查看日志确认失败 workflow。
3. 如果只是网络或模型临时失败，直接重跑 `graphrag index --root . --verbose`。
4. 如果是 prompt 或配置错误，修复配置后重跑；这类变更可能导致 cache key 改变，相关 LLM 调用会重新执行。
5. 如果输出表已经被错误配置污染，再手动清理对应输出目录后重建。

## 4. 为什么不做过度自动续跑

不同 storage provider 对“覆盖、追加、幂等写入”的语义不同。为了避免把损坏的中间表继续传播，本仓库只增强进度和缓存复用，不在底层强行跳过已有输出表。

更严格的断点续跑可以后续按 workflow 粒度实现：记录每个 workflow 的输入配置哈希、输出表状态和完成标记，只有三者完全匹配时才跳过该 workflow。
