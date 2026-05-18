# Copyright (C) 2026 Microsoft
# Licensed under the MIT License

"""Streaming text embedding operation."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from graphrag_llm.tokenizer import Tokenizer
from graphrag_storage.tables.table import Table
from graphrag_vectors import VectorStore, VectorStoreDocument

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.index.operations.embed_text.run_embed_text import run_embed_text
from graphrag.logger.progress import Progress

if TYPE_CHECKING:
    from graphrag_llm.embedding import LLMEmbedding

logger = logging.getLogger(__name__)


async def embed_text(
    input_table: Table,
    callbacks: WorkflowCallbacks,
    model: "LLMEmbedding",
    tokenizer: Tokenizer,
    embed_column: str,
    batch_size: int,
    batch_max_tokens: int,
    num_threads: int,
    vector_store: VectorStore,
    id_column: str = "id",
    output_table: Table | None = None,
) -> int:
    """Embed text from a streaming Table into a vector store.

    Rows are buffered before flushing to ``run_embed_text``,
    which dispatches API batches concurrently up to
    ``num_threads``.  The buffer is sized so each flush produces
    enough batches to saturate the concurrency limit.

    每次 flush 后都会通过 callbacks 上报已写入向量库的行数。
    这样长时间建库时可以观察进度；如果中途失败，日志中的已完成行数也能帮助
    判断需要从哪个 workflow 或哪类 embedding 重新跑。
    """
    vector_store.create_index()

    buffer: list[dict[str, Any]] = []
    total_rows = 0
    # 正常配置下 batch_size 和 num_threads 都应为正数。
    # 这里加下限是为了避免错误配置导致 flush_size=0，从而每读一行都触发刷新。
    flush_size = max(1, batch_size * max(1, num_threads))

    async for row in input_table:
        text = row.get(embed_column)
        if text is None:
            text = ""

        buffer.append({
            id_column: row[id_column],
            embed_column: text,
        })

        if len(buffer) >= flush_size:
            flushed = await _flush_embedding_buffer(
                buffer,
                embed_column,
                id_column,
                callbacks,
                model,
                tokenizer,
                batch_size,
                batch_max_tokens,
                num_threads,
                vector_store,
                output_table,
            )
            total_rows += flushed
            _report_embedding_progress(callbacks, embed_column, total_rows)
            buffer.clear()

    if buffer:
        flushed = await _flush_embedding_buffer(
            buffer,
            embed_column,
            id_column,
            callbacks,
            model,
            tokenizer,
            batch_size,
            batch_max_tokens,
            num_threads,
            vector_store,
            output_table,
        )
        total_rows += flushed
        _report_embedding_progress(callbacks, embed_column, total_rows)

    return total_rows


def _report_embedding_progress(
    callbacks: WorkflowCallbacks,
    embed_column: str,
    completed_rows: int,
) -> None:
    """Report completed embedding rows without pretending to know the final total.

    当前输入表是流式读取的，Table 接口不保证能提前拿到总行数。
    ConsoleWorkflowCallbacks 又需要 total_items 才能计算百分比，所以这里把
    total_items 设置为当前已完成数：含义是“已经稳定写入了多少行”，不是全局百分比。
    """
    callbacks.progress(
        Progress(
            description=f"embed {embed_column} completed rows: ",
            total_items=completed_rows,
            completed_items=completed_rows,
        )
    )


async def _flush_embedding_buffer(
    buffer: list[dict[str, Any]],
    embed_column: str,
    id_column: str,
    callbacks: WorkflowCallbacks,
    model: "LLMEmbedding",
    tokenizer: Tokenizer,
    batch_size: int,
    batch_max_tokens: int,
    num_threads: int,
    vector_store: VectorStore,
    output_table: Table | None,
) -> int:
    """Embed a buffer of rows and load results into the vector store."""
    texts: list[str] = [row[embed_column] for row in buffer]
    ids: list[str] = [row[id_column] for row in buffer]

    result = await run_embed_text(
        texts,
        callbacks,
        model,
        tokenizer,
        batch_size,
        batch_max_tokens,
        num_threads,
    )

    vectors = result.embeddings or []
    skipped = 0
    documents: list[VectorStoreDocument] = []
    for doc_id, doc_vector in zip(ids, vectors, strict=True):
        if doc_vector is None:
            skipped += 1
            continue
        if type(doc_vector) is np.ndarray:
            doc_vector = doc_vector.tolist()
        documents.append(
            VectorStoreDocument(
                id=doc_id,
                vector=doc_vector,
            )
        )

    vector_store.load_documents(documents)

    if skipped > 0:
        logger.warning(
            "Skipped %d rows with None embeddings out of %d",
            skipped,
            len(buffer),
        )

    if output_table is not None:
        for doc_id, doc_vector in zip(ids, vectors, strict=True):
            if doc_vector is None:
                continue
            if type(doc_vector) is np.ndarray:
                doc_vector = doc_vector.tolist()
            await output_table.write({"id": doc_id, "embedding": doc_vector})

    return len(buffer)
