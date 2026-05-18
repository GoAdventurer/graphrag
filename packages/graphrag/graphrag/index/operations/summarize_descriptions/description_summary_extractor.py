# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'SummarizationResult' and 'SummarizeExtractor' models."""

import json
import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from graphrag.index.typing.error_handler import ErrorHandlerFn

if TYPE_CHECKING:
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.types import LLMCompletionResponse

# 下面这些 key 会被用于 prompt.format(...)
# prompt 模板中通过 {entity_name}、{description_list}、{max_length} 占位。
ENTITY_NAME_KEY = "entity_name"
DESCRIPTION_LIST_KEY = "description_list"
MAX_LENGTH_KEY = "max_length"

logger = logging.getLogger(__name__)


@dataclass
class SummarizationResult:
    """Unipartite graph extraction result class definition."""

    # 实体摘要时 id 是实体 title；关系摘要时 id 是 (source, target)。
    id: str | tuple[str, str]
    # LLM 合并后的最终描述。
    description: str


class SummarizeExtractor:
    """Unipartite graph extractor class definition."""

    _model: "LLMCompletion"
    _summarization_prompt: str
    _on_error: ErrorHandlerFn
    _max_summary_length: int
    _max_input_tokens: int

    def __init__(
        self,
        model: "LLMCompletion",
        max_summary_length: int,
        max_input_tokens: int,
        summarization_prompt: str,
        on_error: ErrorHandlerFn | None = None,
    ):
        """Init method definition."""
        # TODO: streamline construction
        # model 是 LLMCompletion 实例，负责实际调用大模型。
        self._model = model
        # 使用模型自带 tokenizer 估算 prompt 和描述列表的 token 数，避免超过输入上限。
        self._tokenizer = model.tokenizer
        # 描述总结 prompt 模板。
        self._summarization_prompt = summarization_prompt
        self._on_error = on_error or (lambda _e, _s, _d: None)
        # 输出摘要的最大长度，由配置 summarize_descriptions.max_length 控制。
        self._max_summary_length = max_summary_length
        # 输入 prompt 的最大 token 数，用于控制一次塞给 LLM 的描述数量。
        self._max_input_tokens = max_input_tokens

    async def __call__(
        self,
        id: str | tuple[str, str],
        descriptions: list[str],
    ) -> SummarizationResult:
        """Call method definition."""
        result = ""
        if len(descriptions) == 0:
            # 没有描述时返回空字符串。
            result = ""
        elif len(descriptions) == 1:
            # 只有一条描述时不需要调用 LLM，直接复用原描述，节省成本。
            result = descriptions[0]
        elif _should_use_local_summary(descriptions):
            # 少量短描述直接在本地合并即可。
            # 这类内容通常已经是结构化短句，调用 CodeBuddy 反而可能因为排队或卡顿拖慢建库。
            result = _summarize_descriptions_locally(descriptions)
        else:
            # 多条描述需要合并去重、压缩表达，因此调用 LLM 总结。
            result = await self._summarize_descriptions(id, descriptions)

        return SummarizationResult(
            id=id,
            description=result or "",
        )

    async def _summarize_descriptions(
        self, id: str | tuple[str, str], descriptions: list[str]
    ) -> str:
        """Summarize descriptions into a single description."""
        # 关系 id 可能是 tuple/list，为了让 prompt 中展示稳定，先做排序处理。
        sorted_id = sorted(id) if isinstance(id, list) else id

        # Safety check, should always be a list
        if not isinstance(descriptions, list):
            descriptions = [descriptions]

        # Sort description lists
        if len(descriptions) > 1:
            # 排序让相同输入在多次运行中保持稳定顺序，有利于缓存和可复现。
            descriptions = sorted(descriptions)

        # Iterate over descriptions, adding all until the max input tokens is reached
        # 先预留 prompt 模板自身占用的 token，再逐条加入 description。
        usable_tokens = self._max_input_tokens - self._tokenizer.num_tokens(
            self._summarization_prompt
        )
        descriptions_collected = []
        result = ""

        for i, description in enumerate(descriptions):
            # 每加入一条描述，就扣除它占用的 token。
            usable_tokens -= self._tokenizer.num_tokens(description)
            descriptions_collected.append(description)

            # If buffer is full, or all descriptions have been added, summarize
            if (usable_tokens < 0 and len(descriptions_collected) > 1) or (
                i == len(descriptions) - 1
            ):
                # Calculate result (final or partial)
                # 如果描述太多超出 token 限制，就先对当前批次做一次部分总结。
                result = await self._summarize_descriptions_with_llm(
                    sorted_id, descriptions_collected
                )

                # If we go for another loop, reset values to new
                if i != len(descriptions) - 1:
                    # 将部分总结结果作为下一轮输入的第一条描述，
                    # 继续合并剩余描述，形成“滚动压缩”。
                    descriptions_collected = [result]
                    usable_tokens = (
                        self._max_input_tokens
                        - self._tokenizer.num_tokens(self._summarization_prompt)
                        - self._tokenizer.num_tokens(result)
                    )

        return result

    async def _summarize_descriptions_with_llm(
        self, id: str | tuple[str, str] | list[str], descriptions: list[str]
    ):
        """Summarize descriptions using the LLM."""
        # 将实体/关系标识、描述列表和最大长度填入 prompt，
        # 让 LLM 输出一段去重、完整、长度受控的最终描述。
        prompt = self._summarization_prompt.format(**{
            ENTITY_NAME_KEY: json.dumps(id, ensure_ascii=False),
            DESCRIPTION_LIST_KEY: json.dumps(sorted(descriptions), ensure_ascii=False),
            MAX_LENGTH_KEY: self._max_summary_length,
        })
        try:
            response: LLMCompletionResponse = await self._model.completion_async(
                messages=prompt,
            )  # type: ignore
            # Calculate result
            # GraphRAG 只取 LLM 返回的文本内容作为最终 description。
            return response.content
        except Exception as e:  # pragma: no cover - fallback path depends on provider
            logger.warning(
                "description summarization failed; using local fallback",
                exc_info=e,
            )
            self._on_error(
                e,
                traceback.format_exc(),
                {
                    "id": id,
                    "description_count": len(descriptions),
                },
            )
            return _summarize_descriptions_locally(descriptions)


def _should_use_local_summary(descriptions: list[str]) -> bool:
    """Return True when deterministic merging is enough for description text."""
    total_chars = sum(len(description) for description in descriptions)
    return len(descriptions) <= 5 and total_chars <= 500


def _summarize_descriptions_locally(descriptions: list[str]) -> str:
    """Merge short descriptions without an LLM call."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for description in descriptions:
        text = str(description).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text.rstrip("。.;；"))
    if not cleaned:
        return ""
    return "；".join(cleaned) + "。"
