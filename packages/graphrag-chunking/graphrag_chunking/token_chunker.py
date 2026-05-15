# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'TokenChunker' class."""

from collections.abc import Callable
from typing import Any

from graphrag_chunking.chunker import Chunker
from graphrag_chunking.create_chunk_results import create_chunk_results
from graphrag_chunking.text_chunk import TextChunk


class TokenChunker(Chunker):
    """A chunker that splits text into token-based chunks."""

    def __init__(
        self,
        size: int,
        overlap: int,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
        **kwargs: Any,
    ) -> None:
        """Create a token chunker instance."""
        # size 控制每个 chunk 最多包含多少 token。
        self._size = size
        # overlap 控制相邻 chunk 之间重复保留多少 token，用来保留上下文连续性。
        self._overlap = overlap
        # encode/decode 由外部 tokenizer 提供，确保切片长度和模型实际 token 计算一致。
        self._encode = encode
        self._decode = decode

    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None
    ) -> list[TextChunk]:
        """Chunk the text into token-based chunks."""
        # 先按 token 数切成字符串列表，再包装为 TextChunk，
        # TextChunk 会记录 chunk 文本、序号、字符范围和 token 数等元信息。
        chunks = split_text_on_tokens(
            text,
            chunk_size=self._size,
            chunk_overlap=self._overlap,
            encode=self._encode,
            decode=self._decode,
        )
        return create_chunk_results(chunks, transform=transform, encode=self._encode)


def split_text_on_tokens(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
) -> list[str]:
    """Split a single text and return chunks using the tokenizer."""
    result = []
    # 将整篇文档一次性 token 化，后面所有切片位置都基于 token 下标计算。
    # 这意味着切片边界是模型视角下的 token 边界，而不是自然段或句子边界。
    input_tokens = encode(text)

    # 第一个窗口从 0 开始，长度最多为 chunk_size。
    start_idx = 0
    cur_idx = min(start_idx + chunk_size, len(input_tokens))
    chunk_tokens = input_tokens[start_idx:cur_idx]

    while start_idx < len(input_tokens):
        # 将当前 token 窗口还原成文本，作为一个 chunk。
        chunk_text = decode(list(chunk_tokens))
        result.append(chunk_text)  # Append chunked text as string
        if cur_idx == len(input_tokens):
            break
        # 下一片向前移动 chunk_size - chunk_overlap。
        # 例如 size=1200、overlap=100，则下一片从当前起点 +1100 开始，
        # 因此前后两片会共享 100 个 token。
        start_idx += chunk_size - chunk_overlap
        cur_idx = min(start_idx + chunk_size, len(input_tokens))
        chunk_tokens = input_tokens[start_idx:cur_idx]

    return result
