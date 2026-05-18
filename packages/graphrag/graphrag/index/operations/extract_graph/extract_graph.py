# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing extract_graph method."""

import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.enums import AsyncType
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.utils.derive_from_rows import derive_from_rows

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion

logger = logging.getLogger(__name__)


async def extract_graph(
    text_units: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    text_column: str,
    id_column: str,
    model: "LLMCompletion",
    prompt: str,
    entity_types: list[str],
    max_gleanings: int,
    num_threads: int,
    async_type: AsyncType,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract a graph from a piece of text using a language model."""
    num_started = 0

    async def run_strategy(row):
        nonlocal num_started
        # 从当前 text_unit 行里取出文本和 ID。
        # text 会作为 LLM 输入，id 会作为 source_id 记录实体/关系来自哪个切片。
        text = row[text_column]
        id = row[id_column]
        # 对单个 text_unit 执行实体/关系抽取。
        # 返回值是两个 DataFrame：当前切片抽到的实体表和关系表。
        result = await _run_extract_graph(
            text=text,
            source_id=id,
            entity_types=entity_types,
            model=model,
            prompt=prompt,
            max_gleanings=max_gleanings,
        )
        num_started += 1
        return result

    # derive_from_rows 会遍历 text_units 的每一行，并按配置并发执行 run_strategy。
    # 这里的 num_threads / async_type 控制 LLM 抽取的并发方式。
    results = await derive_from_rows(
        text_units,
        run_strategy,
        callbacks,
        num_threads=num_threads,
        async_type=async_type,
        progress_msg="extract graph progress: ",
    )

    # results 是每个 text_unit 的抽取结果列表：
    # [
    #   (entities_df_for_chunk_1, relationships_df_for_chunk_1),
    #   (entities_df_for_chunk_2, relationships_df_for_chunk_2),
    #   ...
    # ]
    entity_dfs = []
    relationship_dfs = []
    for result in results:
        if result:
            entity_dfs.append(result[0])
            relationship_dfs.append(result[1])

    # 将所有切片的实体表合成一个全局实体表，并按 title/type 聚合去重。
    entities = _merge_entities(entity_dfs)
    # 将所有切片的关系表合成一个全局关系表，并按 source/target 聚合去重。
    relationships = _merge_relationships(relationship_dfs)
    logger.info(
        "Merged extracted graph: chunks=%d, entities=%d, relationships=%d",
        len(results),
        len(entities),
        len(relationships),
    )
    # 过滤孤儿关系：如果一条关系的 source 或 target 没有对应实体，则移除。
    # 这是对 LLM 输出的兜底清洗，避免后续图构建出现断边。
    relationships = filter_orphan_relationships(relationships, entities)

    return (entities, relationships)


async def _run_extract_graph(
    text: str,
    source_id: str,
    entity_types: list[str],
    model: "LLMCompletion",
    prompt: str,
    max_gleanings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the graph intelligence entity extraction strategy."""
    # GraphExtractor 封装了：
    # - prompt 填充
    # - LLM 调用
    # - 多轮 gleaning 补抽
    # - LLM 输出解析为实体/关系 DataFrame
    extractor = GraphExtractor(
        model=model,
        prompt=prompt,
        max_gleanings=max_gleanings,
        on_error=lambda e, s, d: logger.error(
            "Entity Extraction Error", exc_info=e, extra={"stack": s, "details": d}
        ),
    )
    text = text.strip()

    # 对当前切片执行抽取。
    # source_id 会写入每条实体/关系记录，用于后续追溯来源 text_unit。
    entities_df, relationships_df = await extractor(
        text,
        entity_types=entity_types,
        source_id=source_id,
    )

    return (entities_df, relationships_df)


def _merge_entities(entity_dfs) -> pd.DataFrame:
    if not entity_dfs:
        return pd.DataFrame(
            columns=["title", "type", "description", "text_unit_ids", "frequency"]
        )

    # 将多个 text_unit 的实体 DataFrame 上下拼接成一张大表。
    # ignore_index=True 表示重新生成连续行号，不保留每个小表原来的 index。
    all_entities = pd.concat(entity_dfs, ignore_index=True)
    # LLM 在不同 chunk 中可能把同一实体输出成轻微不同的形式，例如：
    # "Open AI" / "OPENAI"、"张 三" / "张三"。
    # 这里先做确定性的规范化，再按规范化名称聚合，避免后续图谱出现重复节点。
    all_entities["title"] = all_entities["title"].map(_canonical_entity_title)
    all_entities["type"] = all_entities["type"].fillna("UNKNOWN").map(str)
    return (
        all_entities
        # 按实体名称 title 和实体类型 type 聚合。
        # 同名但类型不同的实体会被保留为不同实体，例如 APPLE/ORG 和 APPLE/PRODUCT。
        .groupby(["title", "type"], sort=False)
        .agg(
            # 收集同一实体在不同切片里的所有描述，后续会再交给 LLM 总结。
            description=("description", list),
            # 收集这个实体出现在哪些 text_unit 中，方便后续追溯证据来源。
            text_unit_ids=("source_id", _deduplicate_preserve_order),
            # 统计实体被抽到多少次，作为频率特征。
            frequency=("source_id", "count"),
        )
        # groupby 后 title/type 会变成索引，reset_index 将它们恢复为普通列。
        .reset_index()
    )


def _merge_relationships(relationship_dfs) -> pd.DataFrame:
    if not relationship_dfs:
        return pd.DataFrame(
            columns=["source", "target", "description", "text_unit_ids", "weight"]
        )

    # 将多个 text_unit 的关系 DataFrame 上下拼接成一张大表。
    all_relationships = pd.concat(relationship_dfs, ignore_index=False)
    # 与实体归并保持一致，先把关系两端实体名做规范化。
    # 这样 filter_orphan_relationships 才能稳定判断关系端点是否存在。
    all_relationships["source"] = all_relationships["source"].map(_canonical_entity_title)
    all_relationships["target"] = all_relationships["target"].map(_canonical_entity_title)
    # weight 理论上由 LLM 输出，但实际可能缺失、为空或是字符串。
    # 统一转成数字，无法解析时按 1.0 处理，表示“这条关系至少出现过一次”。
    if "weight" not in all_relationships.columns:
        all_relationships["weight"] = 1.0
    else:
        all_relationships["weight"] = pd.to_numeric(
            all_relationships["weight"],
            errors="coerce",
        ).fillna(1.0)
    return (
        all_relationships
        # 按关系两端聚合。source 和 target 相同的关系被认为是同一条边。
        .groupby(["source", "target"], sort=False)
        .agg(
            # 收集同一条边在不同切片中的所有描述，后续由 LLM 合并总结。
            description=("description", list),
            # 收集这条关系来自哪些 text_unit。
            text_unit_ids=("source_id", _deduplicate_preserve_order),
            # 将 LLM 给出的关系强度累加，作为边权重。
            weight=("weight", "sum"),
        )
        .reset_index()
    )


def _canonical_entity_title(value) -> str:
    """Normalize LLM entity names before merging.

    这里故意只做“确定性、低风险”的规范化：
    - 去掉首尾空白；
    - 统一大写，兼容英文实体；
    - 移除空白和常见包裹符号，减少格式差异；
    - 不做语义别名合并，例如“北京”和“北京市”不会强行合并。

    真正的同义实体合并可以后续接 LLM/词典别名表，但不能在这里凭猜测合并，
    否则容易把两个不同实体误合成一个节点。
    """
    text = str(value or "").strip()
    text = re.sub(r"[\"'`“”‘’]+", "", text)
    text = _normalize_entity_spacing(text)
    text = text.upper()
    return text


def _deduplicate_preserve_order(values) -> list:
    """Deduplicate source ids while keeping first-seen order."""
    return list(dict.fromkeys(values))


def _normalize_entity_spacing(text: str) -> str:
    """Normalize spacing without destroying meaningful English names.

    中文实体里偶尔会出现“张 三”这类被 LLM 拆开的空格，可以安全合并；
    英文实体的空格通常是名称本身的一部分，例如 NEW YORK、OPEN AI，
    只能压缩连续空格，不能全部删除。
    """
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    return text
