# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from graphrag_llm.completion import create_completion

from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader
from graphrag.index.operations.extract_graph.extract_graph import (
    extract_graph as extractor,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to create the base entity graph."""
    logger.info("Workflow started: extract_graph")
    # DataReader 是对 output_table_provider 的一层读取封装。
    # 前一个 workflow 已经把原始文档切成 text_units，本阶段从这里读取切片表。
    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()

    # 读取实体/关系抽取使用的大模型配置。
    # 具体用哪个模型由配置中的 extract_graph.completion_model_id 决定。
    extraction_model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    # resolved_prompts 会读取 prompt 文件或默认 prompt，
    # 其中 extraction_prompt 会要求 LLM 按固定格式输出 entity/relationship。
    extraction_prompts = config.extract_graph.resolved_prompts()
    # 创建用于抽取实体/关系的 LLMCompletion 实例。
    # cache 使用独立子命名空间，避免和其他 LLM 调用混在一起。
    extraction_model = create_completion(
        extraction_model_config,
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # 读取实体/关系描述总结使用的大模型配置。
    # 它可以和抽取模型相同，也可以在配置中单独指定更便宜或更强的模型。
    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    # summarize_prompt 用来把多个切片中抽出的零散描述合并成一段清晰描述。
    summarization_prompts = config.summarize_descriptions.resolved_prompts()
    # 创建用于总结实体/关系描述的 LLMCompletion 实例。
    summarization_model = create_completion(
        summarization_model_config,
        cache=context.cache.child(config.summarize_descriptions.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    # 执行完整图谱抽取流程：
    # 1. 对每个 text_unit 调用 LLM 抽实体和关系
    # 2. 合并重复实体/关系
    # 3. 保留 raw_* 快照数据
    # 4. 再调用 LLM 总结实体/关系描述
    entities, relationships, raw_entities, raw_relationships = await extract_graph(
        text_units=text_units,
        callbacks=context.callbacks,
        extraction_model=extraction_model,
        extraction_prompt=extraction_prompts.extraction_prompt,
        entity_types=config.extract_graph.entity_types,
        max_gleanings=config.extract_graph.max_gleanings,
        extraction_num_threads=config.concurrent_requests,
        extraction_async_type=config.async_mode,
        summarization_model=summarization_model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        summarization_prompt=summarization_prompts.summarize_prompt,
        summarization_num_threads=config.concurrent_requests,
    )

    # 写入正式实体表和关系表。
    # 后续 finalize_graph、create_communities、community_reports 都会读取这两张表。
    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)

    # 如果配置打开 raw_graph，则额外保存“未总结前”的原始抽取结果。
    # 这对调试 LLM 抽取质量很有用，可以看到模型最初返回了哪些描述。
    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe(
            "raw_entities", raw_entities
        )
        await context.output_table_provider.write_dataframe(
            "raw_relationships", raw_relationships
        )

    logger.info("Workflow completed: extract_graph")
    return WorkflowFunctionOutput(
        result={
            "entities": entities,
            "relationships": relationships,
        }
    )


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    extraction_model: "LLMCompletion",
    extraction_prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    extraction_num_threads: int,
    extraction_async_type: AsyncType,
    summarization_model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    summarization_num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All the steps to create the base entity graph."""
    # extractor 是底层抽取操作：
    # 它会遍历 text_units，每个切片都生成一份局部实体表和局部关系表，
    # 然后按实体名/关系两端合并成全局 extracted_entities / extracted_relationships。
    extracted_entities, extracted_relationships = await extractor(
        text_units=text_units,
        callbacks=callbacks,
        text_column="text",
        id_column="id",
        model=extraction_model,
        prompt=extraction_prompt,
        entity_types=entity_types,
        max_gleanings=max_gleanings,
        num_threads=extraction_num_threads,
        async_type=extraction_async_type,
    )

    # 如果没有抽到实体，GraphRAG 后续无法构图，因此直接失败。
    if len(extracted_entities) == 0:
        error_msg = "Graph Extraction failed. No entities detected during extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 如果没有抽到关系，也无法形成图的边，因此直接失败。
    if len(extracted_relationships) == 0:
        error_msg = (
            "Graph Extraction failed. No relationships detected during extraction."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 在描述总结前保存一份原始结果。
    # raw_entities/raw_relationships 保留的是“多个描述组成 list”的中间态，
    # 方便后续对比总结前后的变化。
    raw_entities = extracted_entities.copy()
    raw_relationships = extracted_relationships.copy()

    # 对实体和关系的 description 做二次总结。
    # 例如同一个实体在多个切片中出现，会有多条描述；
    # 这里会用 LLM 合并成更完整、去重后的单条描述。
    entities, relationships = await get_summarized_entities_relationships(
        extracted_entities=extracted_entities,
        extracted_relationships=extracted_relationships,
        callbacks=callbacks,
        model=summarization_model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        summarization_prompt=summarization_prompt,
        num_threads=summarization_num_threads,
    )

    return (entities, relationships, raw_entities, raw_relationships)


async def get_summarized_entities_relationships(
    extracted_entities: pd.DataFrame,
    extracted_relationships: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: "LLMCompletion",
    max_summary_length: int,
    max_input_tokens: int,
    summarization_prompt: str,
    num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the entities and relationships."""
    # summarize_descriptions 会分别处理实体描述和关系描述：
    # - entity_summaries: 按 title 输出实体摘要
    # - relationship_summaries: 按 source/target 输出关系摘要
    entity_summaries, relationship_summaries = await summarize_descriptions(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=callbacks,
        model=model,
        max_summary_length=max_summary_length,
        max_input_tokens=max_input_tokens,
        prompt=summarization_prompt,
        num_threads=num_threads,
    )

    # 关系表先移除原来的 description 列，再按 source/target 合并总结后的 description。
    # 这样最终 relationships 中的 description 是 LLM 总结后的干净描述。
    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )

    # 实体表同理：移除原来的 description list，
    # 再按 title 合并总结后的 description。
    extracted_entities.drop(columns=["description"], inplace=True)
    entities = extracted_entities.merge(entity_summaries, on="title", how="left")
    return entities, relationships
