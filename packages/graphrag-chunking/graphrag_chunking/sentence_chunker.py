# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'SentenceChunker' class."""

from collections.abc import Callable
from typing import Any

import nltk

from graphrag_chunking.bootstrap_nltk import bootstrap
from graphrag_chunking.chunker import Chunker
from graphrag_chunking.create_chunk_results import create_chunk_results
from graphrag_chunking.text_chunk import TextChunk


class SentenceChunker(Chunker):
    """A chunker that splits text into sentence-based chunks."""

    def __init__(
        self, encode: Callable[[str], list[int]] | None = None, **kwargs: Any
    ) -> None:
        """Create a sentence chunker instance."""
        # encode 是可选的；如果传入，会在生成 TextChunk 时计算每个句子的 token 数。
        self._encode = encode
        # 确保 NLTK 的句子切分资源可用，例如 punkt 分句模型。
        bootstrap()

    def chunk(
        self, text: str, transform: Callable[[str], str] | None = None
    ) -> list[TextChunk]:
        """Chunk the text into sentence-based chunks."""
        # 按自然句子边界切分文本。
        # 这种方式语义边界更自然，但每个 chunk 长度不固定，默认建库流程通常使用 token 切片。
        sentences = nltk.sent_tokenize(text.strip())
        results = create_chunk_results(
            sentences, transform=transform, encode=self._encode
        )
        # nltk sentence tokenizer may trim whitespace, so we need to adjust start/end chars
        for index, result in enumerate(results):
            txt = result.text
            start = result.start_char
            # NLTK 分句可能去掉句子前后的空白。
            # 这里重新在原文里查找真实起点，修正 TextChunk 的字符位置。
            actual_start = text.find(txt, start)
            delta = actual_start - start
            if delta > 0:
                result.start_char += delta
                result.end_char += delta
                # bump the next to keep the start check from falling too far behind
                if index < len(results) - 1:
                    results[index + 1].start_char += delta
                    results[index + 1].end_char += delta
        return results
