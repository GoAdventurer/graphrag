# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Graph extraction helpers that return tabular data."""

import logging
import re
import traceback
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag_llm.utils import (
    CompletionMessagesBuilder,
)

from graphrag.index.typing.error_handler import ErrorHandlerFn
from graphrag.index.utils.string import clean_str
from graphrag.prompts.index.extract_graph import (
    CONTINUE_PROMPT,
    LOOP_PROMPT,
)

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse

INPUT_TEXT_KEY = "input_text"
RECORD_DELIMITER_KEY = "record_delimiter"
COMPLETION_DELIMITER_KEY = "completion_delimiter"
ENTITY_TYPES_KEY = "entity_types"
# LLM 输出中，同一条记录内部字段的分隔符。
# 例如实体记录格式：
# ("entity"<|><entity_name><|><entity_type><|><entity_description>)
TUPLE_DELIMITER = "<|>"
# LLM 输出中，不同实体/关系记录之间的分隔符。
RECORD_DELIMITER = "##"
# LLM 输出结束标记，用来告诉解析器后面没有更多记录。
COMPLETION_DELIMITER = "<|COMPLETE|>"

logger = logging.getLogger(__name__)


class GraphExtractor:
    """Unipartite graph extractor class definition."""

    _model: "LLMCompletion"
    _extraction_prompt: str
    _max_gleanings: int
    _on_error: ErrorHandlerFn

    def __init__(
        self,
        model: "LLMCompletion",
        prompt: str,
        max_gleanings: int,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # model 是统一封装后的 LLM 客户端，负责真正发起 completion 请求。
        self._model = model
        # extraction_prompt 是实体/关系抽取提示词模板，里面会填充 input_text 和 entity_types。
        self._extraction_prompt = prompt
        # max_gleanings 控制最多补抽几轮。
        # GraphRAG 认为 LLM 第一轮可能漏掉实体/关系，所以可以继续追问补充。
        self._max_gleanings = max_gleanings
        # 抽取失败时的错误处理回调。默认什么都不做，外层 workflow 会继续处理其他切片。
        self._on_error = on_error or (lambda _e, _s, _d: None)

    async def __call__(
        self, text: str, entity_types: list[str], source_id: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract entities and relationships from the supplied text."""
        try:
            # 对单个 text_unit 调用 LLM，得到原始字符串形式的实体/关系列表。
            result = await self._process_document(text, entity_types)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.exception("error extracting graph")
            self._on_error(
                e,
                traceback.format_exc(),
                {
                    "source_id": source_id,
                    "text": text,
                },
            )
            # 单个切片失败时返回空表，避免一个坏切片中断整个建库流程。
            return _empty_entities_df(), _empty_relationships_df()

        # 将 LLM 返回的固定格式字符串解析为两个 DataFrame：
        # entities_df 和 relationships_df。
        return self._process_result(
            result,
            source_id,
            TUPLE_DELIMITER,
            RECORD_DELIMITER,
        )

    async def _process_document(self, text: str, entity_types: list[str]) -> str:
        # 构造第一轮 LLM 消息。
        # prompt 中会注入当前切片文本 input_text，以及允许抽取的实体类型 entity_types。
        messages_builder = CompletionMessagesBuilder().add_user_message(
            self._extraction_prompt.format(**{
                INPUT_TEXT_KEY: text,
                ENTITY_TYPES_KEY: ",".join(entity_types),
            })
        )

        # 第一次抽取：让 LLM 从当前 text_unit 中直接抽取实体和关系。
        response: LLMCompletionResponse = await self._model.completion_async(
            messages=messages_builder.build(),
        )  # type: ignore
        results = response.content
        # 把模型回答加入上下文，后续补抽时 LLM 能看到自己已经抽过什么。
        messages_builder.add_assistant_message(results)

        # if gleanings are specified, enter a loop to extract more entities
        # there are two exit criteria: (a) we hit the configured max, (b) the model says there are no more entities
        if self._max_gleanings > 0:
            for i in range(self._max_gleanings):
                # 继续要求 LLM 补充遗漏的实体/关系，并要求使用同样的输出格式。
                messages_builder.add_user_message(CONTINUE_PROMPT)
                response: LLMCompletionResponse = await self._model.completion_async(
                    messages=messages_builder.build(),
                )  # type: ignore
                response_text = response.content
                messages_builder.add_assistant_message(response_text)
                # 多轮补抽结果直接拼接到第一轮结果后面，后续统一解析。
                results += response_text

                # if this is the final glean, don't bother updating the continuation flag
                if i >= self._max_gleanings - 1:
                    break

                # 询问 LLM 是否仍有遗漏。
                # 如果模型回答不是 Y，就提前停止补抽，降低调用成本。
                messages_builder.add_user_message(LOOP_PROMPT)
                response: LLMCompletionResponse = await self._model.completion_async(
                    messages=messages_builder.build(),
                )  # type: ignore
                if response.content != "Y":
                    break

        return results

    def _process_result(
        self,
        result: str,
        source_id: str,
        tuple_delimiter: str,
        record_delimiter: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parse the result string into entity and relationship data frames."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # LLM 按 prompt 要求用 ## 分隔多条记录。
        # 每条记录可能是一条 entity，也可能是一条 relationship。
        records = [r.strip() for r in result.split(record_delimiter)]

        for raw_record in records:
            # 去掉记录最外层的小括号，方便按 <|> 拆字段。
            record = re.sub(r"^\(|\)$", "", raw_record.strip())
            if not record or record == COMPLETION_DELIMITER:
                continue

            # 记录内部使用 <|> 分隔字段。
            # entity:      "entity", name, type, description
            # relationship:"relationship", source, target, description, weight
            record_attributes = record.split(tuple_delimiter)
            record_type = record_attributes[0]

            if record_type == '"entity"' and len(record_attributes) >= 4:
                # 实体名和实体类型统一转大写，减少同一实体因大小写不同造成的重复。
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])
                # source_id 记录实体来自哪个 text_unit，方便后续追溯证据来源。
                entities.append({
                    "title": entity_name,
                    "type": entity_type,
                    "description": entity_description,
                    "source_id": source_id,
                })

            if record_type == '"relationship"' and len(record_attributes) >= 5:
                # 关系的 source/target 必须对应实体名。
                # 后续流程会过滤掉 source/target 不在实体表中的孤儿关系。
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                try:
                    # relationship_strength 会作为边权重 weight。
                    weight = float(record_attributes[-1])
                except ValueError:
                    # 如果模型没有返回合法数字，给一个默认权重，避免解析失败。
                    weight = 1.0

                relationships.append({
                    "source": source,
                    "target": target,
                    "description": edge_description,
                    "source_id": source_id,
                    "weight": weight,
                })

        # 统一返回 DataFrame，哪怕没有抽到内容也返回带固定列名的空表。
        # 这样外层合并多个切片结果时，不需要处理 None 或缺列问题。
        entities_df = pd.DataFrame(entities) if entities else _empty_entities_df()
        relationships_df = (
            pd.DataFrame(relationships) if relationships else _empty_relationships_df()
        )

        return entities_df, relationships_df


def _empty_entities_df() -> pd.DataFrame:
    # 空实体表的标准 schema。
    return pd.DataFrame(columns=["title", "type", "description", "source_id"])


def _empty_relationships_df() -> pd.DataFrame:
    # 空关系表的标准 schema。
    return pd.DataFrame(
        columns=["source", "target", "weight", "description", "source_id"]
    )
