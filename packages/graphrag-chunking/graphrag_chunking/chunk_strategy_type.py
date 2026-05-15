# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Chunk strategy type enumeration."""

from enum import StrEnum


class ChunkerType(StrEnum):
    """ChunkerType class definition."""

    # 按 token 固定窗口切片。GraphRAG 默认使用这种策略，
    # 因为 LLM/Embedding 模型的上下文限制通常也是按 token 计算。
    Tokens = "tokens"
    # 按句子边界切片。它更符合自然语言边界，但切片大小不稳定，
    # 在长文档批量建库时通常不如 token 窗口容易控制成本和长度。
    Sentence = "sentence"
