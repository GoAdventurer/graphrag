# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, ConfigDict, Field

from graphrag_chunking.chunk_strategy_type import ChunkerType


class ChunkingConfig(BaseModel):
    """Configuration section for chunking."""

    # 允许配置文件里出现额外字段。
    # 这样外部项目可以给自定义 chunker 传入自己的参数，而不会被 Pydantic 拦截。
    model_config = ConfigDict(extra="allow")
    """Allow extra fields to support custom cache implementations."""

    # 切片策略类型。默认 tokens，表示按 tokenizer 的 token 数量切片。
    # 也可以配置为 sentence，表示按句子边界切片。
    type: str = Field(
        description="The chunking type to use.",
        default=ChunkerType.Tokens,
    )
    # tokenizer 使用的编码模型。
    # GraphRAG 会根据这个模型拿到 encode/decode 方法，用于把文本和 token ID 相互转换。
    encoding_model: str | None = Field(
        description="The encoding model to use.",
        default=None,
    )
    # 每个 chunk 的目标大小，默认 1200 个 token。
    # 这是 GraphRAG 建库时控制上下文粒度的关键参数：太小会割裂语义，太大检索不够精细。
    size: int = Field(
        description="The chunk size to use.",
        default=1200,
    )
    # 相邻 chunk 之间保留的重叠 token 数，默认 100。
    # overlap 用来降低“重要信息刚好被切断”的风险，让前后切片有连续上下文。
    overlap: int = Field(
        description="The chunk overlap to use.",
        default=100,
    )
    # 可选：把源文档的某些元数据字段追加到每个 chunk 前面。
    # 例如 title、creation_date，可以让每个切片自带来源上下文，方便后续抽取和检索。
    prepend_metadata: list[str] | None = Field(
        description="Metadata fields from the source document to prepend on each chunk.",
        default=None,
    )
