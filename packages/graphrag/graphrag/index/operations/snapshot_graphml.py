# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing snapshot_graphml method definition."""

import networkx as nx
import pandas as pd
from graphrag_storage import Storage


async def snapshot_graphml(
    edges: pd.DataFrame,
    name: str,
    storage: Storage,
) -> None:
    """Take a entire snapshot of a graph to standard graphml format."""
    # relationships 表就是图里的边表：
    # - source / target 会被 networkx 识别为边的两个端点
    # - weight 会作为边属性写入 GraphML
    # 这里不会重新做实体抽取，只是把已经整理好的关系表转换成可视化工具能读取的 GraphML。
    graph = nx.from_pandas_edgelist(edges, edge_attr=["weight"])
    # 将 networkx 图对象序列化成 GraphML 文本。
    # GraphML 可以被 Gephi、Cytoscape、yEd 等图分析/可视化工具打开。
    graphml = "\n".join(nx.generate_graphml(graph))
    # storage 是 GraphRAG 的输出存储抽象。
    # 默认 file storage 时会写成本地文件 graph.graphml；
    # 如果换成 blob/cosmos 等 provider，则会写到对应远端存储。
    await storage.set(name + ".graphml", graphml)
