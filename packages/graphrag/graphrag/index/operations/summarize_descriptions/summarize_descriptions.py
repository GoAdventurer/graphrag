# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing the summarize_descriptions verb."""

import asyncio
import logging
from typing import TYPE_CHECKING

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.summarize_descriptions.description_summary_extractor import (
    SummarizeExtractor,
)
from graphrag.index.operations.summarize_descriptions.typing import (
    SummarizedDescriptionResult,
)
from graphrag.logger.progress import ProgressTicker, progress_ticker

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def summarize_descriptions(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
    num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize entity and relationship descriptions from an entity graph, using a language model."""

    async def get_summarized(
        nodes: pd.DataFrame, edges: pd.DataFrame, semaphore: asyncio.Semaphore
    ):
        # 总进度 = 实体数量 + 关系数量。
        # 每个实体/关系都会独立生成一条最终 description。
        ticker_length = len(nodes) + len(edges)

        ticker = progress_ticker(
            callbacks.progress,
            ticker_length,
            description="Summarize entity/relationship description progress: ",
        )

        # 为每个实体创建一个总结任务。
        # row.description 在前一步是 list[str]，包含该实体从多个 text_unit 中抽到的所有描述。
        # sorted(set(...)) 用于去重并保证输入顺序稳定。
        node_futures = [
            do_summarize_descriptions(
                str(row.title),  # type: ignore
                sorted(set(row.description)),  # type: ignore
                ticker,
                semaphore,
            )
            for row in nodes.itertuples(index=False)
        ]

        # 并发执行所有实体描述总结任务。
        node_results = await asyncio.gather(*node_futures)

        # 将 SummarizedDescriptionResult 转成实体摘要 DataFrame 所需的行结构。
        node_descriptions = [
            {
                "title": result.id,
                "description": result.description,
            }
            for result in node_results
        ]

        # 为每条关系创建一个总结任务。
        # 关系用 (source, target) 作为唯一标识，描述来自多个 text_unit 的同一条边。
        edge_futures = [
            do_summarize_descriptions(
                (str(row.source), str(row.target)),  # type: ignore
                sorted(set(row.description)),  # type: ignore
                ticker,
                semaphore,
            )
            for row in edges.itertuples(index=False)
        ]

        # 并发执行所有关系描述总结任务。
        edge_results = await asyncio.gather(*edge_futures)

        # 将关系摘要结果转成 relationship_summaries DataFrame。
        edge_descriptions = [
            {
                "source": result.id[0],
                "target": result.id[1],
                "description": result.description,
            }
            for result in edge_results
        ]

        entity_descriptions = pd.DataFrame(node_descriptions)
        relationship_descriptions = pd.DataFrame(edge_descriptions)
        return entity_descriptions, relationship_descriptions

    async def do_summarize_descriptions(
        id: str | tuple[str, str],
        descriptions: list[str],
        ticker: ProgressTicker,
        semaphore: asyncio.Semaphore,
    ):
        # semaphore 用来限制并发 LLM 请求数量，避免同时发起过多请求导致限流或资源压力。
        async with semaphore:
            results = await run_summarize_descriptions(
                id,
                descriptions,
                model,
                max_summary_length,
                max_input_tokens,
                prompt,
            )
            # 每完成一个实体/关系摘要，更新一次进度。
            ticker(1)
        return results

    # num_threads 控制描述总结阶段的最大并发数。
    semaphore = asyncio.Semaphore(num_threads)

    return await get_summarized(entities_df, relationships_df, semaphore)


async def run_summarize_descriptions(
    id: str | tuple[str, str],
    descriptions: list[str],
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
) -> SummarizedDescriptionResult:
    """Run the graph intelligence entity extraction strategy."""
    # SummarizeExtractor 负责真正调用 LLM 合并多条描述。
    # 外层函数只负责并发调度和 DataFrame 形态转换。
    extractor = SummarizeExtractor(
        model=model,
        summarization_prompt=prompt,
        on_error=lambda e, stack, details: logger.error(
            "Entity Extraction Error",
            exc_info=e,
            extra={"stack": stack, "details": details},
        ),
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
    )

    # id 可以是实体 title，也可以是关系 (source, target)。
    result = await extractor(id=id, descriptions=descriptions)
    return SummarizedDescriptionResult(id=result.id, description=result.description)
