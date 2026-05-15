# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The TextChunk dataclass."""

from dataclasses import dataclass


@dataclass
class TextChunk:
    """Result of chunking a document."""

    # 切片器直接切出来的原始文本，不包含额外追加的元数据。
    original: str
    """Raw original text chunk before any transformation."""

    # 最终用于建库的文本。它可能等于 original，
    # 也可能经过 transform 追加了 title、creation_date 等文档元数据。
    text: str
    """The final text content of this chunk."""

    # 当前 chunk 在同一篇文档内的顺序编号，从 0 开始。
    index: int
    """Zero-based index of this chunk within the source document."""

    # 当前 chunk 在原始文档中的起始字符位置。
    start_char: int
    """Character index where the raw chunk text begins in the source document."""

    # 当前 chunk 在原始文档中的结束字符位置。
    end_char: int
    """Character index where the raw chunk text ends in the source document."""

    # 当前 chunk 的 token 数。如果创建结果时没有传 tokenizer，则保持 None。
    token_count: int | None = None
    """Number of tokens in the final chunk text, if computed."""
