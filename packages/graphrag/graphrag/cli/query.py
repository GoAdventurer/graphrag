# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI implementation of the query subcommand."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

import graphrag.api as api
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.config.enums import SearchMethod
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader

if TYPE_CHECKING:
    import pandas as pd

# ruff: noqa: T201


def run_auto_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    dynamic_community_selection: bool,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Route a query to the most suitable search strategy.

    GraphRAG 原生提供 local/global/drift/basic 四种检索方式，但原 CLI
    需要用户手动选择。这里增加一层轻量级自动路由：
    - 不调用 LLM，避免查询前额外消耗 token 和时间；
    - 只根据问题措辞做确定性判断，便于调试和复现；
    - 选择后仍然复用原来的检索函数，避免复制检索实现。
    """
    method, reason = _select_auto_search_method(query)
    if verbose:
        print(f"[auto] selected method={method.value}: {reason}")

    if method is SearchMethod.GLOBAL:
        return run_global_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            query=query,
            verbose=verbose,
        )
    if method is SearchMethod.DRIFT:
        return run_drift_search(
            data_dir=data_dir,
            root_dir=root_dir,
            community_level=community_level,
            response_type=response_type,
            streaming=streaming,
            query=query,
            verbose=verbose,
        )
    if method is SearchMethod.BASIC:
        return run_basic_search(
            data_dir=data_dir,
            root_dir=root_dir,
            response_type=response_type,
            streaming=streaming,
            query=query,
            verbose=verbose,
        )
    return run_local_search(
        data_dir=data_dir,
        root_dir=root_dir,
        community_level=community_level,
        response_type=response_type,
        streaming=streaming,
        query=query,
        verbose=verbose,
    )


def _select_auto_search_method(query: str) -> tuple[SearchMethod, str]:
    """Choose a search method with transparent keyword heuristics.

    这是“工程可控”的第一版自动路由。它不会假装理解所有语义，而是把常见
    问题意图分成四类：
    - global：全局概览、主题总结、整体趋势；
    - drift：需要围绕实体做分析推理、原因解释、影响链路；
    - basic：强调原文、引用、证据片段的朴素文本检索；
    - local：默认的实体/关系精确问答。

    后续如果需要更强的智能路由，可以把这个函数替换成 LLM 分类器，但 CLI
    层和各检索函数不需要改变。
    """
    normalized = query.strip().lower()

    basic_keywords = (
        "原文",
        "引用",
        "出处",
        "来源",
        "证据",
        "片段",
        "chunk",
        "quote",
        "cite",
        "source",
        "evidence",
    )
    if any(keyword in normalized for keyword in basic_keywords):
        return (SearchMethod.BASIC, "question asks for source text or evidence")

    global_keywords = (
        "总结",
        "概览",
        "整体",
        "全局",
        "主题",
        "趋势",
        "有哪些",
        "分别是什么",
        "overview",
        "summarize",
        "summary",
        "theme",
        "trend",
        "overall",
    )
    if any(keyword in normalized for keyword in global_keywords):
        return (SearchMethod.GLOBAL, "question is broad and benefits from community reports")

    drift_keywords = (
        "为什么",
        "原因",
        "影响",
        "关联路径",
        "链路",
        "推理",
        "分析",
        "对比",
        "区别",
        "怎么导致",
        "why",
        "impact",
        "compare",
        "difference",
        "reason",
        "analyze",
    )
    if any(keyword in normalized for keyword in drift_keywords):
        return (SearchMethod.DRIFT, "question needs entity-centered reasoning")

    return (SearchMethod.LOCAL, "default to entity and relationship grounded search")


def print_context_summary(context_data: Any, max_items: int = 5) -> None:
    """Print a compact explanation of the retrieved context.

    Query API 返回的 context_data 往往包含若干张表或列表。默认不打印是为了
    保持回答干净；当 CLI 传入 --show-context 时，这里只打印“用了哪些上下文
    和大概多少条”，避免把完整检索上下文刷屏。
    """
    if not context_data:
        print("\n[context] no context data captured")
        return

    print("\n[context] retrieved context summary")
    if isinstance(context_data, dict):
        for name, value in context_data.items():
            _print_context_value(name, value, max_items)
        return

    _print_context_value("context", context_data, max_items)


def _print_context_value(name: str, value: Any, max_items: int) -> None:
    """Print one context object without exposing large content."""
    if hasattr(value, "shape"):
        rows, cols = value.shape
        print(f"- {name}: dataframe rows={rows}, columns={cols}")
        return

    if isinstance(value, list):
        print(f"- {name}: list items={len(value)}")
        for item in value[:max_items]:
            print(f"  sample: {_context_preview(item)}")
        return

    if isinstance(value, dict):
        print(f"- {name}: dict keys={list(value.keys())[:max_items]}")
        return

    print(f"- {name}: {type(value).__name__}")


def _context_preview(item: Any) -> str:
    """Return a single-line preview for context diagnostics."""
    if isinstance(item, dict):
        keys = list(item.keys())[:5]
        return "{" + ", ".join(f"{key}={str(item.get(key))[:40]!r}" for key in keys) + "}"
    text = str(item).replace("\n", " ")
    return text[:120]


def run_global_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int | None,
    dynamic_community_selection: bool,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a global search with a given query.

    Loads index files required for global search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "entities",
            "communities",
            "community_reports",
        ],
        optional_list=[],
    )

    entities: pd.DataFrame = dataframe_dict["entities"]
    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.global_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.global_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_local_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a local search with a given query.

    Loads index files required for local search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        optional_list=[
            "covariates",
        ],
    )

    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    text_units: pd.DataFrame = dataframe_dict["text_units"]
    relationships: pd.DataFrame = dataframe_dict["relationships"]
    entities: pd.DataFrame = dataframe_dict["entities"]
    covariates: pd.DataFrame | None = dataframe_dict["covariates"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.local_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                covariates=covariates,
                community_level=community_level,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=community_level,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_drift_search(
    data_dir: Path | None,
    root_dir: Path,
    community_level: int,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a local search with a given query.

    Loads index files required for local search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
    )

    communities: pd.DataFrame = dataframe_dict["communities"]
    community_reports: pd.DataFrame = dataframe_dict["community_reports"]
    text_units: pd.DataFrame = dataframe_dict["text_units"]
    relationships: pd.DataFrame = dataframe_dict["relationships"]
    entities: pd.DataFrame = dataframe_dict["entities"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.drift_search_streaming(
                config=config,
                entities=entities,
                communities=communities,
                community_reports=community_reports,
                text_units=text_units,
                relationships=relationships,
                community_level=community_level,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())

    # not streaming
    response, context_data = asyncio.run(
        api.drift_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            community_level=community_level,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def run_basic_search(
    data_dir: Path | None,
    root_dir: Path,
    response_type: str,
    streaming: bool,
    query: str,
    verbose: bool,
):
    """Perform a basics search with a given query.

    Loads index files required for basic search and calls the Query API.
    """
    cli_overrides: dict[str, Any] = {}
    if data_dir:
        cli_overrides["output_storage"] = {"base_dir": str(data_dir)}
    config = load_config(
        root_dir=root_dir,
        cli_overrides=cli_overrides,
    )

    dataframe_dict = _resolve_output_files(
        config=config,
        output_list=[
            "text_units",
        ],
    )

    text_units: pd.DataFrame = dataframe_dict["text_units"]

    if streaming:

        async def run_streaming_search():
            full_response = ""
            context_data = {}

            def on_context(context: Any) -> None:
                nonlocal context_data
                context_data = context

            callbacks = NoopQueryCallbacks()
            callbacks.on_context = on_context

            async for stream_chunk in api.basic_search_streaming(
                config=config,
                text_units=text_units,
                response_type=response_type,
                query=query,
                callbacks=[callbacks],
                verbose=verbose,
            ):
                full_response += stream_chunk
                print(stream_chunk, end="")
                sys.stdout.flush()
            print()
            return full_response, context_data

        return asyncio.run(run_streaming_search())
    # not streaming
    response, context_data = asyncio.run(
        api.basic_search(
            config=config,
            text_units=text_units,
            response_type=response_type,
            query=query,
            verbose=verbose,
        )
    )
    print(response)

    return response, context_data


def _resolve_output_files(
    config: GraphRagConfig,
    output_list: list[str],
    optional_list: list[str] | None = None,
) -> dict[str, Any]:
    """Read indexing output files to a dataframe dict, with correct column types."""
    dataframe_dict = {}
    storage_obj = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage_obj)
    reader = DataReader(table_provider)
    for name in output_list:
        df_value = asyncio.run(getattr(reader, name)())
        dataframe_dict[name] = df_value

    # for optional output files, set the dict entry to None instead of erroring out if it does not exist
    if optional_list:
        for optional_file in optional_list:
            file_exists = asyncio.run(table_provider.has(optional_file))
            if file_exists:
                df_value = asyncio.run(getattr(reader, optional_file)())
                dataframe_dict[optional_file] = df_value
            else:
                dataframe_dict[optional_file] = None
    return dataframe_dict
