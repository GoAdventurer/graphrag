# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LLMCompletion implementation backed by the local CodeBuddy CLI."""

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.completion.completion import LLMCompletion
from graphrag_llm.types import (
    LLMChoiceChunk,
    LLMChoiceDelta,
    LLMCompletionChunk,
)
from graphrag_llm.utils import (
    create_completion_response,
    structure_completion_response,
)

if TYPE_CHECKING:
    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsStore
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        LLMCompletionArgs,
        LLMCompletionChunk,
        LLMCompletionMessagesParam,
        LLMCompletionResponse,
        ResponseFormat,
    )


DEFAULT_CODEBUDDY_CMD = "/root/.local/bin/codebuddy"
DEFAULT_CODEBUDDY_TIMEOUT = 120


class CodeBuddyCompletion(LLMCompletion):
    """LLMCompletion implementation that calls CodeBuddy as a local CLI."""

    _model_config: "ModelConfig"
    _metrics_store: "MetricsStore"
    _tokenizer: "Tokenizer"
    _codebuddy_cmd: str
    _timeout: int

    def __init__(
        self,
        *,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        codebuddy_cmd: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CodeBuddyCompletion.

        Extra config fields supported in YAML:
        - codebuddy_cmd: path to the CodeBuddy executable.
        - timeout: subprocess timeout in seconds.
        """
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._codebuddy_cmd = (
            codebuddy_cmd
            or os.getenv("CODEBUDDY_CMD")
            or DEFAULT_CODEBUDDY_CMD
        )
        self._timeout = int(
            timeout
            or os.getenv("CODEBUDDY_TIMEOUT", str(DEFAULT_CODEBUDDY_TIMEOUT))
        )

    def completion(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | Iterator[LLMCompletionChunk]":
        """Sync completion method."""
        messages = kwargs.pop("messages")
        response_format = kwargs.pop("response_format", None)
        stream = kwargs.pop("stream", False)

        prompt = _messages_to_prompt(messages)
        output = self._call_codebuddy(prompt)
        if stream:
            # GraphRAG 的查询链路会用 stream=True 消费 LLM，即使 CLI 没有显式开启
            # streaming。CodeBuddy CLI 当前返回完整文本，因此这里把完整结果包装成
            # 一个单 chunk 的流，保持接口兼容。
            return _single_chunk_iterator(output, self._model_config.model)

        response = create_completion_response(output)

        if response_format is not None:
            structured_response = structure_completion_response(
                response.content, response_format
            )
            response.formatted_response = structured_response

        return response

    async def completion_async(
        self,
        /,
        **kwargs: Unpack["LLMCompletionArgs[ResponseFormat]"],
    ) -> "LLMCompletionResponse[ResponseFormat] | AsyncIterator[LLMCompletionChunk]":
        """Async completion method."""
        if kwargs.get("stream", False):
            messages = kwargs.pop("messages")
            kwargs.pop("response_format", None)
            kwargs.pop("stream", None)
            prompt = _messages_to_prompt(messages)
            output = await asyncio.to_thread(self._call_codebuddy, prompt)
            return _single_chunk_async_iterator(output, self._model_config.model)

        return await asyncio.to_thread(self.completion, **kwargs)  # type: ignore

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store."""
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer."""
        return self._tokenizer

    def _call_codebuddy(self, prompt: str) -> str:
        """Call CodeBuddy CLI and return plain text output."""
        command = [
            self._codebuddy_cmd,
            "--model",
            self._model_config.model,
            "-y",
            "-p",
            prompt,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self._timeout,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(details or "CodeBuddy 调用失败")

        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            raise RuntimeError("CodeBuddy 未返回内容")
        return output


def _messages_to_prompt(messages: "LLMCompletionMessagesParam") -> str:
    """Convert GraphRAG/OpenAI style messages into a single CodeBuddy prompt."""
    if isinstance(messages, str):
        return messages

    prompt_parts: list[str] = []
    for message in _ensure_sequence(messages):
        role = str(_get_message_value(message, "role") or "user")
        content = _get_message_value(message, "content")
        prompt_parts.append(f"{role}: {_stringify_content(content)}")
    return "\n\n".join(prompt_parts)


def _single_chunk_iterator(content: str, model: str) -> Iterator[LLMCompletionChunk]:
    """Yield a single OpenAI-compatible streaming chunk."""
    yield _create_stream_chunk(content, model)


async def _single_chunk_async_iterator(
    content: str,
    model: str,
) -> AsyncIterator[LLMCompletionChunk]:
    """Yield a single OpenAI-compatible streaming chunk asynchronously."""
    yield _create_stream_chunk(content, model)


def _create_stream_chunk(content: str, model: str) -> LLMCompletionChunk:
    """Create a ChatCompletionChunk carrying the full CodeBuddy response."""
    return LLMCompletionChunk(
        id="codebuddy-completion-chunk",
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=[
            LLMChoiceChunk(
                index=0,
                delta=LLMChoiceDelta(content=content),
                finish_reason="stop",
            )
        ],
    )


def _ensure_sequence(messages: "LLMCompletionMessagesParam") -> Sequence[Any]:
    """Return messages as a sequence for type checkers."""
    return messages if isinstance(messages, Sequence) else [messages]


def _get_message_value(message: Any, key: str) -> Any:
    """Read a message field from either dict-like or object-like messages."""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _stringify_content(content: Any) -> str:
    """Convert OpenAI content shapes into plain text for the CLI prompt."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text") or item.get("content") or item))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)
