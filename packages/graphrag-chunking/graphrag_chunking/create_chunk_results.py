# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'create_chunk_results' function."""

from collections.abc import Callable

from graphrag_chunking.text_chunk import TextChunk


def create_chunk_results(
    chunks: list[str],
    transform: Callable[[str], str] | None = None,
    encode: Callable[[str], list[int]] | None = None,
) -> list[TextChunk]:
    """Create chunk results from a list of text chunks. The index assignments are 0-based and assume chunks were not stripped relative to the source text."""
    results = []
    # start_char/end_char 用来标记 chunk 在原始文本中的字符位置。
    # 对 token 切片来说，因为 decode 后文本可能和原文边界不完全一一对应，这里是近似顺序位置；
    # 对 sentence 切片，SentenceChunker 后面会再修正真实起点。
    start_char = 0
    for index, chunk in enumerate(chunks):
        end_char = start_char + len(chunk) - 1  # 0-based indices
        result = TextChunk(
            original=chunk,
            # transform 可以给 chunk 追加元数据，例如标题、创建时间等。
            # original 保留原始切片文本，text 是最终写入索引的文本。
            text=transform(chunk) if transform else chunk,
            index=index,
            start_char=start_char,
            end_char=end_char,
        )
        if encode:
            # 如果提供 tokenizer，则记录最终文本的 token 数，便于后续控制上下文长度。
            result.token_count = len(encode(result.text))
        results.append(result)
        start_char = end_char + 1
    return results
