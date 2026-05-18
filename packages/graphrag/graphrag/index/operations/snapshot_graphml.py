# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing snapshot_graphml method definition."""

import json

import networkx as nx
import pandas as pd
from graphrag_storage import Storage


async def snapshot_graphml(
    edges: pd.DataFrame,
    name: str,
    storage: Storage,
    nodes: pd.DataFrame | None = None,
) -> None:
    """Take a entire snapshot of a graph to standard graphml format."""
    # relationships 表就是图里的边表，source / target 会被 networkx 识别为边的两个端点。
    # 这里不会重新做实体抽取，只是把已经整理好的实体/关系表转换成可视化工具能读取的 GraphML。
    edge_table = _sanitize_graphml_table(edges)
    edge_attrs = [
        column
        for column in [
            "id",
            "human_readable_id",
            "description",
            "weight",
            "combined_degree",
        ]
        if column in edge_table.columns
    ]
    graph = nx.from_pandas_edgelist(
        edge_table,
        source="source",
        target="target",
        edge_attr=edge_attrs,
    )

    if nodes is not None and not nodes.empty:
        node_table = _sanitize_graphml_table(nodes)
        node_attrs = [
            column
            for column in [
                "id",
                "human_readable_id",
                "type",
                "description",
                "frequency",
                "degree",
            ]
            if column in node_table.columns
        ]
        # GraphML 节点属性会跟节点一起写入文件。
        # 可视化时可以按 type 上色、按 degree 调整大小、点击节点查看 description。
        for row in node_table.to_dict(orient="records"):
            title = row.get("title")
            if title is None:
                continue
            graph.add_node(title, **{attr: row.get(attr, "") for attr in node_attrs})

    # 将 networkx 图对象序列化成 GraphML 文本。
    # GraphML 可以被 Gephi、Cytoscape、yEd 等图分析/可视化工具打开。
    graphml = "\n".join(nx.generate_graphml(graph))
    # storage 是 GraphRAG 的输出存储抽象。
    # 默认 file storage 时会写成本地文件 graph.graphml；
    # 如果换成 blob/cosmos 等 provider，则会写到对应远端存储。
    await storage.set(name + ".graphml", graphml)
    # 同步生成一个自包含 HTML，方便在 Cursor 中直接预览、缩放和搜索图谱。
    await storage.set(name + ".html", _generate_graph_html(graph))


def _sanitize_graphml_table(data: pd.DataFrame) -> pd.DataFrame:
    """Convert dataframe values to GraphML-safe scalar attributes.

    NetworkX 的 GraphML writer 只支持字符串、数字、布尔值这类标量。
    GraphRAG 的中间表里经常有 list/dict（例如 text_unit_ids），直接写入会报错。
    所以这里在导出前做一次保守清洗，保证可视化文件稳定生成。
    """
    safe = data.copy()
    for column in _GRAPHML_NUMERIC_COLUMNS:
        if column in safe.columns:
            safe[column] = pd.to_numeric(safe[column], errors="coerce").fillna(0)
    for column in safe.columns:
        safe[column] = safe[column].map(_sanitize_graphml_value)
    return safe


_GRAPHML_NUMERIC_COLUMNS = {
    "human_readable_id",
    "weight",
    "combined_degree",
    "frequency",
    "degree",
}


def _sanitize_graphml_value(value):
    """Return a scalar value accepted by GraphML."""
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple | set):
        return ", ".join(str(item) for item in value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list | tuple | set):
            return ", ".join(str(item) for item in converted)
        return str(converted)
    if isinstance(value, dict):
        return str(value)
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return ""
    return str(value)


def _generate_graph_html(graph: nx.Graph) -> str:
    """Generate a self-contained interactive HTML preview for Cursor."""
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        degree = int(attrs.get("degree") or graph.degree(node_id) or 1)
        nodes.append({
            "id": str(node_id),
            "label": str(node_id),
            "type": str(attrs.get("type", "")),
            "description": str(attrs.get("description", "")),
            "degree": degree,
            "size": min(34, 8 + degree * 1.5),
        })

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            "source": str(source),
            "target": str(target),
            "weight": float(attrs.get("weight") or 1),
            "description": str(attrs.get("description", "")),
        })

    payload = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GraphRAG 图谱预览</title>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e5e7eb; }}
    #toolbar {{ position: fixed; left: 16px; top: 16px; z-index: 2; display: flex; gap: 8px; align-items: center; padding: 10px; background: rgba(15, 23, 42, .86); border: 1px solid rgba(148, 163, 184, .35); border-radius: 10px; backdrop-filter: blur(8px); }}
    #search {{ width: 260px; padding: 8px 10px; border-radius: 8px; border: 1px solid #475569; background: #020617; color: #e5e7eb; }}
    button {{ padding: 8px 10px; border-radius: 8px; border: 1px solid #475569; background: #1e293b; color: #e5e7eb; cursor: pointer; }}
    button:hover {{ background: #334155; }}
    #stats {{ font-size: 12px; color: #cbd5e1; }}
    #details-toggle {{ position: fixed; right: 16px; top: 16px; z-index: 3; }}
    #info {{ position: fixed; right: 16px; top: 62px; bottom: 16px; z-index: 2; width: 360px; overflow: auto; padding: 14px; background: rgba(15, 23, 42, .9); border: 1px solid rgba(148, 163, 184, .35); border-radius: 10px; backdrop-filter: blur(8px); }}
    #info.hidden {{ display: none; }}
    #info h2 {{ margin: 0 0 8px; font-size: 18px; }}
    #info .muted {{ color: #94a3b8; font-size: 12px; }}
    #info p {{ line-height: 1.55; white-space: pre-wrap; }}
    #canvas {{ width: 100vw; height: 100vh; display: block; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <input id="search" placeholder="搜索节点名称..." />
    <button id="fit">居中</button>
    <button id="pause">暂停布局</button>
    <span id="stats"></span>
  </div>
  <button id="details-toggle">详情</button>
  <aside id="info" class="hidden"></aside>
  <canvas id="canvas"></canvas>
  <script>
    const graph = {payload};
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const info = document.getElementById('info');
    const search = document.getElementById('search');
    const stats = document.getElementById('stats');
    const fitBtn = document.getElementById('fit');
    const pauseBtn = document.getElementById('pause');
    const detailsToggle = document.getElementById('details-toggle');
    const nodes = graph.nodes;
    const edges = graph.edges;
    const nodeMap = new Map(nodes.map(n => [n.id, n]));
    const palette = ['#38bdf8', '#a78bfa', '#34d399', '#fbbf24', '#fb7185', '#60a5fa', '#f472b6', '#22c55e'];
    const typeColors = new Map();
    let width = 0, height = 0, scale = 1, offsetX = 0, offsetY = 0;
    let running = true, draggingNode = null, panning = false, lastX = 0, lastY = 0, selected = null, query = '';
    let infoVisible = false;

    function colorFor(type) {{
      if (!typeColors.has(type)) typeColors.set(type, palette[typeColors.size % palette.length]);
      return typeColors.get(type);
    }}
    function setInfoVisible(visible) {{
      infoVisible = visible;
      info.classList.toggle('hidden', !visible);
      detailsToggle.textContent = visible ? '隐藏详情' : '详情';
    }}
    function resize() {{
      width = canvas.clientWidth * devicePixelRatio;
      height = canvas.clientHeight * devicePixelRatio;
      canvas.width = width;
      canvas.height = height;
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    }}
    function initPositions() {{
      const radius = Math.min(innerWidth, innerHeight) * 0.34;
      nodes.forEach((n, i) => {{
        const a = Math.PI * 2 * i / Math.max(nodes.length, 1);
        n.x = Math.cos(a) * radius + innerWidth / 2;
        n.y = Math.sin(a) * radius + innerHeight / 2;
        n.vx = 0; n.vy = 0;
      }});
      offsetX = 0; offsetY = 0; scale = 1;
    }}
    function step() {{
      if (!running) return;
      for (const n of nodes) {{
        n.vx += (innerWidth / 2 - n.x) * 0.0008;
        n.vy += (innerHeight / 2 - n.y) * 0.0008;
      }}
      for (let i = 0; i < nodes.length; i++) {{
        for (let j = i + 1; j < nodes.length; j++) {{
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = Math.max(dx * dx + dy * dy, 25);
          const f = 280 / d2;
          a.vx += dx * f; a.vy += dy * f;
          b.vx -= dx * f; b.vy -= dy * f;
        }}
      }}
      for (const e of edges) {{
        const a = nodeMap.get(e.source), b = nodeMap.get(e.target);
        if (!a || !b) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const d = Math.max(Math.hypot(dx, dy), 1);
        const f = (d - 150) * 0.002 * Math.min(e.weight || 1, 5);
        a.vx += dx / d * f; a.vy += dy / d * f;
        b.vx -= dx / d * f; b.vy -= dy / d * f;
      }}
      for (const n of nodes) {{
        if (n === draggingNode) continue;
        n.vx *= 0.82; n.vy *= 0.82;
        n.x += n.vx; n.y += n.vy;
      }}
    }}
    function toWorld(x, y) {{ return {{ x: (x - offsetX) / scale, y: (y - offsetY) / scale }}; }}
    function draw() {{
      ctx.clearRect(0, 0, innerWidth, innerHeight);
      ctx.save();
      ctx.translate(offsetX, offsetY);
      ctx.scale(scale, scale);
      ctx.lineCap = 'round';
      for (const e of edges) {{
        const a = nodeMap.get(e.source), b = nodeMap.get(e.target);
        if (!a || !b) continue;
        ctx.strokeStyle = 'rgba(148, 163, 184, .25)';
        ctx.lineWidth = Math.max(0.7, Math.min(4, (e.weight || 1) / 2));
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }}
      for (const n of nodes) {{
        const matched = query && n.label.toLowerCase().includes(query);
        ctx.fillStyle = colorFor(n.type || 'unknown');
        ctx.strokeStyle = selected === n || matched ? '#ffffff' : 'rgba(15, 23, 42, .9)';
        ctx.lineWidth = selected === n || matched ? 4 : 2;
        ctx.beginPath(); ctx.arc(n.x, n.y, n.size, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        if (scale > 0.45 || selected === n || matched) {{
          ctx.fillStyle = '#e5e7eb';
          ctx.font = '12px sans-serif';
          ctx.fillText(n.label, n.x + n.size + 4, n.y + 4);
        }}
      }}
      ctx.restore();
    }}
    function frame() {{ step(); draw(); requestAnimationFrame(frame); }}
    function nearest(x, y) {{
      const p = toWorld(x, y);
      let best = null, bestD = Infinity;
      for (const n of nodes) {{
        const d = Math.hypot(n.x - p.x, n.y - p.y);
        if (d < n.size + 8 && d < bestD) {{ best = n; bestD = d; }}
      }}
      return best;
    }}
    function showNode(n) {{
      selected = n;
      const related = edges.filter(e => e.source === n.id || e.target === n.id).slice(0, 12);
      info.innerHTML = `<h2>${{n.label}}</h2><div class="muted">类型：${{n.type || '未知'}} | 度：${{n.degree}}</div><p>${{n.description || '暂无描述'}}</p><h3>相关关系</h3>` +
        related.map(e => `<p><b>${{e.source}} → ${{e.target}}</b><br>${{e.description || ''}}</p>`).join('');
      setInfoVisible(true);
    }}
    canvas.addEventListener('mousedown', e => {{
      lastX = e.clientX; lastY = e.clientY;
      draggingNode = nearest(e.clientX, e.clientY);
      panning = !draggingNode;
      if (draggingNode) showNode(draggingNode);
    }});
    canvas.addEventListener('mousemove', e => {{
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (draggingNode) {{
        const p = toWorld(e.clientX, e.clientY);
        draggingNode.x = p.x; draggingNode.y = p.y; draggingNode.vx = 0; draggingNode.vy = 0;
      }} else if (panning) {{
        offsetX += dx; offsetY += dy;
      }}
      lastX = e.clientX; lastY = e.clientY;
    }});
    addEventListener('mouseup', () => {{ draggingNode = null; panning = false; }});
    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const before = toWorld(e.clientX, e.clientY);
      scale *= e.deltaY < 0 ? 1.1 : 0.9;
      scale = Math.max(0.15, Math.min(4, scale));
      offsetX = e.clientX - before.x * scale;
      offsetY = e.clientY - before.y * scale;
    }}, {{ passive: false }});
    search.addEventListener('input', () => {{ query = search.value.trim().toLowerCase(); }});
    fitBtn.onclick = () => {{ initPositions(); }};
    pauseBtn.onclick = () => {{ running = !running; pauseBtn.textContent = running ? '暂停布局' : '继续布局'; }};
    detailsToggle.onclick = () => {{ setInfoVisible(!infoVisible); }};
    stats.textContent = `${{nodes.length}} 个节点 / ${{edges.length}} 条关系`;
    addEventListener('resize', resize);
    resize(); initPositions(); frame();
  </script>
</body>
</html>
"""
