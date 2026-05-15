# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'ChunkerFactory', 'register_chunker', and 'create_chunker'."""

from collections.abc import Callable

from graphrag_common.factory.factory import Factory, ServiceScope

from graphrag_chunking.chunk_strategy_type import ChunkerType
from graphrag_chunking.chunker import Chunker
from graphrag_chunking.chunking_config import ChunkingConfig


class ChunkerFactory(Factory[Chunker]):
    """Factory for creating Chunker instances."""


chunker_factory = ChunkerFactory()


def register_chunker(
    chunker_type: str,
    chunker_initializer: Callable[..., Chunker],
    scope: ServiceScope = "transient",
) -> None:
    """Register a custom chunker implementation.

    Args
    ----
        - chunker_type: str
            The chunker id to register.
        - chunker_initializer: Callable[..., Chunker]
            The chunker initializer to register.
    """
    # 把一个切片器实现注册到工厂中。
    # 后续 create_chunker 只需要根据字符串类型，就能创建对应的切片器实例。
    chunker_factory.register(chunker_type, chunker_initializer, scope)


def create_chunker(
    config: ChunkingConfig,
    encode: Callable[[str], list[int]] | None = None,
    decode: Callable[[list[int]], str] | None = None,
) -> Chunker:
    """Create a chunker implementation based on the given configuration.

    Args
    ----
        - config: ChunkingConfig
            The chunker configuration to use.

    Returns
    -------
        Chunker
            The created chunker implementation.
    """
    # 将 Pydantic 配置转成普通 dict，作为切片器构造参数。
    # config 里包含 size、overlap、type 等切片参数。
    config_model = config.model_dump()
    if encode is not None:
        # token 切片需要 encode：文本 -> token ID 列表。
        config_model["encode"] = encode
    if decode is not None:
        # token 切片需要 decode：token ID 列表 -> 文本。
        config_model["decode"] = decode
    chunker_strategy = config.type

    if chunker_strategy not in chunker_factory:
        # 如果当前策略还没有注册，则按内置策略懒加载对应实现。
        # 这样避免模块加载时提前导入所有切片器，也允许外部注册自定义策略。
        match chunker_strategy:
            case ChunkerType.Tokens:
                from graphrag_chunking.token_chunker import TokenChunker

                register_chunker(ChunkerType.Tokens, TokenChunker)
            case ChunkerType.Sentence:
                from graphrag_chunking.sentence_chunker import SentenceChunker

                register_chunker(ChunkerType.Sentence, SentenceChunker)
            case _:
                msg = f"ChunkingConfig.strategy '{chunker_strategy}' is not registered in the ChunkerFactory. Registered types: {', '.join(chunker_factory.keys())}."
                raise ValueError(msg)

    # 最终返回一个具体切片器实例，例如 TokenChunker 或 SentenceChunker。
    return chunker_factory.create(chunker_strategy, init_args=config_model)
