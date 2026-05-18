#!/usr/bin/env python3
"""Prepare source code repositories as GraphRAG text input."""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INCLUDE_EXTENSIONS = {
    ".go",
    ".md",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".mod",
}

DEFAULT_INCLUDE_NAMES = {
    "makefile",
    "dockerfile",
}

SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "target",
    ".cache",
}

SKIP_FILE_NAMES = {
    "go.sum",
}

LANG_BY_EXTENSION = {
    ".go": "go",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".mod": "go",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a code repository into compact .txt documents for GraphRAG.",
    )
    parser.add_argument("--source", required=True, help="Source code repository path.")
    parser.add_argument("--output", required=True, help="Output directory for .txt docs.")
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=12000,
        help="Maximum source characters included per file document.",
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include truncated source code in each document. Disabled by default to save tokens.",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Write one document per file. By default files are grouped by top-level module.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.exists() or not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    files = list(_iter_source_files(source))
    if args.per_file or args.include_source:
        for idx, path in enumerate(files, 1):
            rel = path.relative_to(source)
            doc = _build_file_document(
                source,
                path,
                args.max_file_chars,
                include_source=args.include_source,
            )
            out_name = f"{idx:04d}_{_safe_name(str(rel))}.txt"
            (output / out_name).write_text(doc, encoding="utf-8")
    else:
        for idx, (module, paths) in enumerate(_group_files_by_module(source, files).items(), 1):
            doc = _build_module_document(source, module, paths)
            out_name = f"{idx:04d}_module_{_safe_name(module)}.txt"
            (output / out_name).write_text(doc, encoding="utf-8")

    overview = _build_project_overview(source, files, output)
    (output / "0000_project_overview.txt").write_text(overview, encoding="utf-8")
    print(f"prepared_code_docs={len(list(output.glob('*.txt')))} output={output}")


def _iter_source_files(source: Path):
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(source).parts[:-1]):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        if path.stat().st_size > 512 * 1024:
            continue
        if _is_supported_file(path):
            yield path


def _is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in DEFAULT_INCLUDE_EXTENSIONS or path.name.lower() in DEFAULT_INCLUDE_NAMES


def _build_project_overview(source: Path, files: list[Path], output: Path) -> str:
    dirs: dict[str, list[str]] = defaultdict(list)
    ext_counter = Counter(path.suffix.lower() or path.name.lower() for path in files)
    for path in files:
        rel = path.relative_to(source)
        directory = str(rel.parent) if str(rel.parent) != "." else "根目录"
        dirs[directory].append(rel.name)

    lines = [
        f"项目名称: {source.name}",
        f"项目路径: {source}",
        f"GraphRAG 代码索引输入目录: {output}",
        "",
        "这是由代码仓库自动生成的 GraphRAG 输入文档，用于建立项目代码图谱。",
        "图谱应重点抽取文件、目录、包、函数、类型、配置、外部依赖、调用关系、测试关系和业务责任边界。",
        "",
        "文件类型统计:",
    ]
    for ext, count in sorted(ext_counter.items()):
        lines.append(f"- {ext}: {count}")

    lines.extend(["", "目录结构摘要:"])
    for directory, names in sorted(dirs.items()):
        preview = ", ".join(names[:20])
        suffix = " ..." if len(names) > 20 else ""
        lines.append(f"- {directory}: {preview}{suffix}")

    return "\n".join(lines) + "\n"


def _group_files_by_module(source: Path, files: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        rel = path.relative_to(source)
        first = rel.parts[0] if len(rel.parts) > 1 else "root"
        grouped[first].append(path)
    return dict(sorted(grouped.items()))


def _build_module_document(source: Path, module: str, paths: list[Path]) -> str:
    lines = [
        f"模块名称: {module}",
        f"项目名称: {source.name}",
        f"模块文件数: {len(paths)}",
        "",
        "这是代码项目的模块级 GraphRAG 输入文档。",
        "请抽取模块、文件、包、函数、类型、配置、依赖、测试、业务职责之间的关系。",
        "",
        "模块文件摘要:",
    ]
    for path in paths:
        lines.extend(_build_file_summary(source, path))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _build_file_summary(source: Path, path: Path) -> list[str]:
    rel = path.relative_to(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    language = LANG_BY_EXTENSION.get(path.suffix.lower(), path.suffix.lstrip(".") or path.name)
    package = _extract_go_package(text) if path.suffix == ".go" else ""
    imports = _extract_go_imports(text) if path.suffix == ".go" else []
    definitions = _extract_go_definitions(text) if path.suffix == ".go" else []
    comments = _extract_go_comments(text) if path.suffix == ".go" else []

    lines = [
        f"- 文件路径: {rel}",
        f"  文件类型: {language}",
    ]
    if package:
        lines.append(f"  Go 包名: {package}")
    if imports:
        lines.append(f"  导入依赖: {', '.join(imports[:30])}")
    if definitions:
        lines.append(f"  代码定义: {', '.join(definitions[:40])}")
    if comments:
        lines.append(f"  关键注释: {'; '.join(comments[:10])}")
    return lines


def _build_file_document(
    source: Path,
    path: Path,
    max_chars: int,
    *,
    include_source: bool,
) -> str:
    rel = path.relative_to(source)
    text = path.read_text(encoding="utf-8", errors="replace")
    language = LANG_BY_EXTENSION.get(path.suffix.lower(), path.suffix.lstrip(".") or path.name)
    package = _extract_go_package(text) if path.suffix == ".go" else ""
    imports = _extract_go_imports(text) if path.suffix == ".go" else []
    definitions = _extract_go_definitions(text) if path.suffix == ".go" else []
    comments = _extract_go_comments(text) if path.suffix == ".go" else []
    truncated = len(text) > max_chars
    content = text[:max_chars]

    lines = [
        f"文件路径: {rel}",
        f"文件类型: {language}",
        f"所属目录: {rel.parent if str(rel.parent) != '.' else '根目录'}",
    ]
    if package:
        lines.append(f"Go 包名: {package}")
    if imports:
        lines.append("导入依赖:")
        lines.extend(f"- {item}" for item in imports)
    if definitions:
        lines.append("代码定义:")
        lines.extend(f"- {item}" for item in definitions)
    if comments:
        lines.append("关键注释:")
        lines.extend(f"- {item}" for item in comments[:20])

    lines.extend([
        "",
        "GraphRAG 抽取提示:",
        "请把该文件抽取为代码图谱中的文件节点、包节点、函数/方法节点、类型节点和依赖节点。",
        "重点保留文件职责、核心函数作用、import 依赖、测试覆盖关系、配置项含义和跨文件调用线索。",
    ])
    if include_source:
        lines.extend([
            "",
            "源码内容:",
            f"```{language}",
            content,
        ])
        if truncated:
            lines.append("\n... 文件过长，后续内容已截断 ...")
        lines.append("```")
    else:
        lines.extend([
            "",
            "源码内容: 已省略。当前文档用于低 token 成本的代码图谱索引，主要依据文件路径、包名、import、定义和注释抽取结构关系。",
        ])
    return "\n".join(lines) + "\n"


def _extract_go_package(text: str) -> str:
    match = re.search(r"(?m)^package\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    return match.group(1) if match else ""


def _extract_go_imports(text: str) -> list[str]:
    imports: list[str] = []
    block = re.search(r"(?ms)^import\s*\((.*?)\)", text)
    if block:
        imports.extend(re.findall(r'"([^"]+)"', block.group(1)))
    imports.extend(re.findall(r'(?m)^import\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?"([^"]+)"', text))
    return sorted(set(imports))


def _extract_go_definitions(text: str) -> list[str]:
    definitions: list[str] = []
    for kind, name in re.findall(r"(?m)^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)\b", text):
        definitions.append(f"{kind}: {name}")
    for receiver, name in re.findall(
        r"(?m)^func\s*(\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text,
    ):
        prefix = "method" if receiver else "func"
        definitions.append(f"{prefix}: {name}")
    return definitions


def _extract_go_comments(text: str) -> list[str]:
    comments: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("//"):
            continue
        comment = stripped.removeprefix("//").strip()
        if comment and len(comment) <= 160:
            comments.append(comment)
    return comments


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return safe[:160].strip("._") or "file"


if __name__ == "__main__":
    main()
