# 本地模型配置：CodeBuddy 与 Ollama

这份文档说明如何把 GraphRAG 的大模型调用切到本地或局域网模型，重点覆盖 CodeBuddy 和 Ollama。这样可以减少外部 API 依赖，也方便在中文资料上反复调试抽取 prompt。

## 1. CodeBuddy 作为 completion provider

本仓库已经新增 `codebuddy` provider。它会通过本机 CodeBuddy CLI 调用模型，适合实体抽取、描述总结、社区摘要和查询回答这些文本生成任务。

示例配置：

```yaml
completion_models:
  default_completion_model:
    type: codebuddy
    model_provider: codebuddy
    model: gpt-5.5
    codebuddy_cmd: /root/.local/bin/codebuddy
    timeout: 600
```

字段说明：

- `type: codebuddy` 表示走本仓库新增的 CodeBuddy completion 实现。
- `model_provider: codebuddy` 保留为提供方标识，便于日志、缓存和配置辨识。
- `model` 会透传给 `codebuddy --model`。
- `codebuddy_cmd` 可选；不配置时默认使用 `/root/.local/bin/codebuddy`，也可以通过环境变量 `CODEBUDDY_CMD` 覆盖。
- `timeout` 可选；脚本默认设置为 600 秒，也可以通过环境变量 `CODEBUDDY_TIMEOUT` 覆盖。GraphRAG 建索引时会批量做实体/关系描述总结，长文本总结可能超过 180 秒。

注意：CodeBuddy 目前只实现了 completion，不实现 embedding。向量化仍需单独配置 embedding provider。

## 2. Ollama 通过 LiteLLM 接入

Ollama 可以走 GraphRAG 原生的 LiteLLM provider。配置里 `model_provider` 写 `ollama`，`model` 写 Ollama 本地模型名，最终会由 LiteLLM 组装成 `ollama/<model>` 调用。

示例配置：

```yaml
completion_models:
  default_completion_model:
    model_provider: ollama
    model: qwen2.5:7b
    api_base: http://localhost:11434
    api_key: ollama
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: ollama
    model: qwen3-embedding:8b
    api_base: http://localhost:11434
    api_key: ollama
    retry:
      type: exponential_backoff

vector_store:
  type: lancedb
  db_uri: output/lancedb
  vector_size: 4096
```

本仓库默认初始化配置已经按上面的 Ollama + 千问设置生成：

- completion 默认：`qwen2.5:7b`
- embedding 默认：`qwen3-embedding:8b`
- embedding 向量维度默认：`4096`

如果你换成其他 embedding 模型，必须同步修改 `vector_store.vector_size`，否则向量库会因为维度不匹配而写入失败。

## 3. 中文资料推荐调整

中文知识库建议同时调整抽取和总结 prompt：

```yaml
extract_graph:
  prompt: "prompts/extract_graph_zh.txt"

summarize_descriptions:
  prompt: "prompts/summarize_descriptions_zh.txt"
```

仓库内提供了中文 prompt 常量：

- `graphrag.prompts.index.extract_graph.GRAPH_EXTRACTION_PROMPT_ZH`
- `graphrag.prompts.index.extract_graph.CONTINUE_PROMPT_ZH`
- `graphrag.prompts.index.extract_graph.LOOP_PROMPT_ZH`
- `graphrag.prompts.index.summarize_descriptions.SUMMARIZE_PROMPT_ZH`

这些模板仍然保留 GraphRAG 的严格输出分隔符，例如 `("entity"<|>...)`、`("relationship"<|>...)`、`##` 和 `<|COMPLETE|>`。不要删除这些符号，否则解析器无法稳定读取实体和关系。

## 4. 查询建议

CLI 默认查询方法已经改为 `auto`。它会根据问题措辞在 `global`、`local`、`drift`、`basic` 之间做确定性路由。

常用命令：

```bash
graphrag query "总结一下这个知识库的主要主题" --root .
graphrag query "张三和星河科技是什么关系" --root . --method local
graphrag query "给我引用相关原文证据" --root . --show-context
```

如果需要严格复现某种检索路径，仍然可以手动指定 `--method global|local|drift|basic`。
