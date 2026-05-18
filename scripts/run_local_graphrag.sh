#!/usr/bin/env bash
set -euo pipefail

# Local GraphRAG runner for Ollama completion and Ollama embeddings.
#
# 常用方式：
#   ./scripts/run_local_graphrag.sh
#   ./scripts/run_local_graphrag.sh chat
#   ./scripts/run_local_graphrag.sh index
#   ./scripts/run_local_graphrag.sh query "张三和星河科技是什么关系"
#
# 可选环境变量：
#   GRAPHRAG_PROJECT_DIR=/data/workspace/AI29_graph_rag/graphrag_demo
#   GRAPHRAG_COMPLETION_PROVIDER=ollama|codebuddy
#   GRAPHRAG_COMPLETION_MODEL=deepseek-r1:14b
#   GRAPHRAG_EMBEDDING_MODEL=qwen3-embedding:8b
#   GRAPHRAG_VECTOR_SIZE=4096
#   GRAPHRAG_SOURCE_FILE=/data/workspace/AI29_graph_rag/graphrag_demo3/red.txt
#   GRAPHRAG_CODE_SOURCE_DIR=/data/workspace/AI29_graph_rag/graphrag_demo2/input/user_asset_list
#   GRAPHRAG_CODE_MAX_FILE_CHARS=12000
#   GRAPHRAG_CODE_INCLUDE_SOURCE=0
#   OLLAMA_API_BASE=http://localhost:11434
#   CODEBUDDY_CMD=/root/.local/bin/codebuddy
#   CODEBUDDY_TIMEOUT=600

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_DIR="${GRAPHRAG_PROJECT_DIR:-$REPO_DIR/graphrag_demo3}"
COMPLETION_PROVIDER="${GRAPHRAG_COMPLETION_PROVIDER:-ollama}"
COMPLETION_MODEL="${GRAPHRAG_COMPLETION_MODEL:-deepseek-r1:14b}"
EMBEDDING_MODEL="${GRAPHRAG_EMBEDDING_MODEL:-qwen3-embedding:8b}"
VECTOR_SIZE="${GRAPHRAG_VECTOR_SIZE:-4096}"
SOURCE_FILE="${GRAPHRAG_SOURCE_FILE:-$PROJECT_DIR/red.txt}"
CODE_SOURCE_DIR="${GRAPHRAG_CODE_SOURCE_DIR:-$PROJECT_DIR/input/user_asset_list}"
CODE_DOCS_DIR="$PROJECT_DIR/input/_graphrag_code_docs"
CODE_MAX_FILE_CHARS="${GRAPHRAG_CODE_MAX_FILE_CHARS:-12000}"
CODE_INCLUDE_SOURCE="${GRAPHRAG_CODE_INCLUDE_SOURCE:-0}"
OLLAMA_API_BASE="${OLLAMA_API_BASE:-http://localhost:11434}"
CODEBUDDY_CMD="${CODEBUDDY_CMD:-/root/.local/bin/codebuddy}"
CODEBUDDY_TIMEOUT="${CODEBUDDY_TIMEOUT:-600}"

ACTION="${1:-chat}"
QUERY_TEXT=""
if [ "$#" -ge 2 ]; then
  QUERY_TEXT="${*:2}"
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "缺少命令：$1" >&2
    exit 1
  fi
}

ensure_ollama_model() {
  local model="$1"
  if ! ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$model"; then
    echo "本机 Ollama 未安装模型：$model" >&2
    echo "请先执行：ollama pull $model" >&2
    exit 1
  fi
}

install_deps() {
  log "安装 GraphRAG 本地依赖"
  cd "$REPO_DIR"
  python -m pip install -r requirements.txt
}

check_ollama() {
  log "检查 Ollama 和 embedding 模型"
  require_command ollama
  ollama list >/dev/null
  ensure_ollama_model "$EMBEDDING_MODEL"
  if [ "$COMPLETION_PROVIDER" = "ollama" ]; then
    ensure_ollama_model "$COMPLETION_MODEL"
  fi
}

check_completion_provider() {
  case "$COMPLETION_PROVIDER" in
    ollama)
      check_ollama
      ;;
    codebuddy)
      log "检查 CodeBuddy 和 Ollama embedding 模型"
      if [ ! -x "$CODEBUDDY_CMD" ]; then
        echo "找不到可执行的 CodeBuddy CLI：$CODEBUDDY_CMD" >&2
        echo "可通过 CODEBUDDY_CMD=/path/to/codebuddy 覆盖。" >&2
        exit 1
      fi
      require_command ollama
      ollama list >/dev/null
      ensure_ollama_model "$EMBEDDING_MODEL"
      ;;
    *)
      echo "不支持的 GRAPHRAG_COMPLETION_PROVIDER：$COMPLETION_PROVIDER" >&2
      echo "可选值：ollama、codebuddy" >&2
      exit 1
      ;;
  esac
}

ensure_demo_input_file() {
  mkdir -p "$PROJECT_DIR/input"
  if [ -d "$CODE_SOURCE_DIR" ]; then
    log "检测到代码项目输入，跳过普通示例文件创建：$CODE_SOURCE_DIR"
    return
  fi
  if [ -f "$SOURCE_FILE" ]; then
    local target_file="$PROJECT_DIR/input/$(basename "$SOURCE_FILE")"
    log "同步 GraphRAG 输入文件并转换为 UTF-8：$SOURCE_FILE -> $target_file"
    SOURCE_FILE="$SOURCE_FILE" TARGET_FILE="$target_file" python - <<'PY'
import os
from pathlib import Path

source = Path(os.environ["SOURCE_FILE"])
target = Path(os.environ["TARGET_FILE"])
data = source.read_bytes()

for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
    try:
        text = data.decode(encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    text = data.decode("utf-8", errors="ignore")

target.write_text(text, encoding="utf-8")
PY
    if [ -f "$PROJECT_DIR/input/test.txt" ]; then
      log "删除旧示例输入文件：$PROJECT_DIR/input/test.txt"
      rm "$PROJECT_DIR/input/test.txt"
    fi
    return
  fi
  if [ ! -f "$PROJECT_DIR/input/test.txt" ]; then
    log "创建示例输入文件：$PROJECT_DIR/input/test.txt"
    cat > "$PROJECT_DIR/input/test.txt" <<'EOF'
张三是星河科技的技术负责人。星河科技正在研发知识图谱和 GraphRAG 系统。
李四负责向量检索模块，张三负责实体关系抽取模块。
星河科技希望通过 GraphRAG 把文档中的实体、关系和社区摘要组织起来，提升问答效果。
EOF
  else
    log "输入文件已存在，保留现有内容：$PROJECT_DIR/input/test.txt"
  fi
}

prepare_code_input_if_needed() {
  if [ ! -d "$CODE_SOURCE_DIR" ]; then
    return
  fi

  log "为代码项目生成 GraphRAG 文本输入：$CODE_SOURCE_DIR"
  local include_source_args=()
  if [ "$CODE_INCLUDE_SOURCE" = "1" ]; then
    include_source_args=(--include-source)
  fi
  python "$REPO_DIR/scripts/prepare_code_graphrag_input.py" \
    --source "$CODE_SOURCE_DIR" \
    --output "$CODE_DOCS_DIR" \
    --max-file-chars "$CODE_MAX_FILE_CHARS" \
    "${include_source_args[@]}"
}

init_project() {
  log "初始化知识库目录：$PROJECT_DIR"
  ensure_demo_input_file
  prepare_code_input_if_needed
  if [ ! -f "$PROJECT_DIR/settings.yaml" ]; then
    # 显式传入 model/embedding，避免 Typer 再弹出交互式 prompt。
    graphrag init \
      --root "$PROJECT_DIR" \
      --model "$COMPLETION_MODEL" \
      --embedding "$EMBEDDING_MODEL"
  else
    log "settings.yaml 已存在，跳过 init"
  fi
  patch_settings
}

patch_settings() {
  log "写入模型配置和中文图谱 prompt：completion=$COMPLETION_PROVIDER, embedding=ollama"
  PROJECT_DIR="$PROJECT_DIR" \
  COMPLETION_PROVIDER="$COMPLETION_PROVIDER" \
  COMPLETION_MODEL="$COMPLETION_MODEL" \
  EMBEDDING_MODEL="$EMBEDDING_MODEL" \
  VECTOR_SIZE="$VECTOR_SIZE" \
  OLLAMA_API_BASE="$OLLAMA_API_BASE" \
  CODEBUDDY_CMD="$CODEBUDDY_CMD" \
  CODEBUDDY_TIMEOUT="$CODEBUDDY_TIMEOUT" \
  python - <<'PY'
import os
from pathlib import Path

import yaml
from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT_ZH
from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT_ZH

project_dir = Path(os.environ["PROJECT_DIR"])
settings_path = project_dir / "settings.yaml"
settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
settings["concurrent_requests"] = 2
code_docs_dir = project_dir / "input" / "_graphrag_code_docs"

# GraphML 的节点名称、实体类型、节点描述、关系描述都来自抽取与描述总结阶段。
# 因此这里直接写入中文 prompt，让后续 index 生成的 parquet/GraphML 天然是中文。
prompts_dir = project_dir / "prompts"
prompts_dir.mkdir(parents=True, exist_ok=True)
(prompts_dir / "extract_graph_zh.txt").write_text(
    GRAPH_EXTRACTION_PROMPT_ZH.strip() + "\n",
    encoding="utf-8",
)
(prompts_dir / "summarize_descriptions_zh.txt").write_text(
    SUMMARIZE_PROMPT_ZH.strip() + "\n",
    encoding="utf-8",
)

settings.setdefault("completion_models", {})
provider = os.environ["COMPLETION_PROVIDER"]
if provider == "codebuddy":
    settings["completion_models"]["default_completion_model"] = {
        "type": "codebuddy",
        "model_provider": "codebuddy",
        "model": os.environ["COMPLETION_MODEL"],
        "codebuddy_cmd": os.environ["CODEBUDDY_CMD"],
        "timeout": int(os.environ["CODEBUDDY_TIMEOUT"]),
        "retry": {"type": "exponential_backoff"},
    }
else:
    settings["completion_models"]["default_completion_model"] = {
        "model_provider": "ollama",
        "model": os.environ["COMPLETION_MODEL"],
        "auth_method": "api_key",
        "api_key": "ollama",
        "api_base": os.environ["OLLAMA_API_BASE"],
        "retry": {"type": "exponential_backoff"},
    }

settings.setdefault("embedding_models", {})
settings["embedding_models"]["default_embedding_model"] = {
    "model_provider": "ollama",
    "model": os.environ["EMBEDDING_MODEL"],
    "auth_method": "api_key",
    "api_key": "ollama",
    "api_base": os.environ["OLLAMA_API_BASE"],
    "retry": {"type": "exponential_backoff"},
}

settings.setdefault("vector_store", {})
settings["vector_store"].update({
    "type": "lancedb",
    "db_uri": "output/lancedb",
    "vector_size": int(os.environ["VECTOR_SIZE"]),
})

settings.setdefault("input", {})
settings["input"].update({
    "type": "text",
    # 代码仓库会先转换为 _graphrag_code_docs/*.txt，再交给 GraphRAG 处理。
    # 这样可以避开 .git、go.sum、大文件和非文本产物，减少噪声与 token 消耗。
    # GraphRAG 读取配置前会做 $ENV 环境变量替换，所以正则里不要使用行尾锚点 "$"。
    "file_pattern": r".*_graphrag_code_docs/.*\.txt" if code_docs_dir.exists() else r".*\.txt",
})

settings.setdefault("extract_graph", {})
settings["extract_graph"].update({
    "completion_model_id": "default_completion_model",
    "prompt": "prompts/extract_graph_zh.txt",
    # 使用中文实体类型，避免 GraphML 中 type 字段继续出现 ORGANIZATION/PERSON 等英文枚举。
    "entity_types": ["组织", "人物", "地点", "事件", "任务", "风险", "系统", "模块", "问题", "规则"],
    "max_gleanings": settings["extract_graph"].get("max_gleanings", 1),
})

settings.setdefault("summarize_descriptions", {})
settings["summarize_descriptions"].update({
    "completion_model_id": "default_completion_model",
    "prompt": "prompts/summarize_descriptions_zh.txt",
    "max_length": settings["summarize_descriptions"].get("max_length", 500),
})

settings_path.write_text(
    yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
PY
}

build_index() {
  log "开始构建 GraphRAG 知识库"
  graphrag index --root "$PROJECT_DIR" --verbose
}

ensure_index_exists() {
  if [ ! -d "$PROJECT_DIR/output" ]; then
    echo "未发现索引目录：$PROJECT_DIR/output" >&2
    echo "请先执行：$0 index" >&2
    exit 1
  fi
}

run_query() {
  if [ -z "${QUERY_TEXT//[[:space:]]/}" ]; then
    echo "query 模式需要传入问题，例如：" >&2
    echo "  $0 query \"张三负责什么\"" >&2
    exit 1
  fi
  ensure_index_exists
  log "执行查询：$QUERY_TEXT"
  graphrag query "$QUERY_TEXT" --root "$PROJECT_DIR" --show-context
}

chat_session() {
  ensure_index_exists
  cd "$PROJECT_DIR"
  log "进入 GraphRAG 会话"
  echo "知识库目录：$PROJECT_DIR"
  echo "输入问题后回车；输入 exit、quit 或 q 退出。"

  while true; do
    printf "\ngraphrag> "
    if ! IFS= read -r question; then
      echo
      break
    fi

    if [ -z "${question//[[:space:]]/}" ]; then
      continue
    fi

    case "$question" in
      exit|quit|q)
        echo "退出 GraphRAG 会话。"
        break
        ;;
    esac

    graphrag query "$question" --root "$PROJECT_DIR" --show-context
  done
}

print_usage() {
  cat <<EOF
用法：
  $0
  $0 chat
  $0 setup
  $0 init
  $0 index
  $0 query "你的问题"
  $0 all

默认使用 Ollama deepseek 做文本生成、Ollama qwen 做 embedding：
  GRAPHRAG_COMPLETION_PROVIDER=ollama \\
  GRAPHRAG_COMPLETION_MODEL=deepseek-r1:14b \\
  $0 all

如需切回 CodeBuddy 做文本生成：
  GRAPHRAG_COMPLETION_PROVIDER=codebuddy \\
  GRAPHRAG_COMPLETION_MODEL=gpt-5.5 \\
  $0 all

代码项目索引：
  如果存在 $PROJECT_DIR/input/user_asset_list，脚本会自动生成
  $PROJECT_DIR/input/_graphrag_code_docs/*.txt 后再建索引。

默认知识库目录：
  $PROJECT_DIR

默认模型：
  completion provider: $COMPLETION_PROVIDER
  completion model:    $COMPLETION_MODEL
  embedding:  $EMBEDDING_MODEL
EOF
}

case "$ACTION" in
  chat)
    check_completion_provider
    init_project
    chat_session
    ;;
  setup)
    install_deps
    check_completion_provider
    ;;
  init)
    check_completion_provider
    init_project
    ;;
  index)
    check_completion_provider
    init_project
    build_index
    ;;
  query)
    check_completion_provider
    init_project
    run_query
    ;;
  all)
    install_deps
    check_completion_provider
    init_project
    build_index
    chat_session
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    echo "未知操作：$ACTION" >&2
    print_usage
    exit 1
    ;;
esac
