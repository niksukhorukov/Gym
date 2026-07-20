#!/usr/bin/env bash
# Run a local Gym baseline against a vLLM-compatible model endpoint.
#
# By default this starts vLLM on CUDA device 1 and runs a one-row MCQA smoke
# rollout with Qwen/Qwen3.5-0.8B. Pass --benchmark or --environment for larger
# baseline runs.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/queue/run_gym_baseline_job.sh [options]

Default smoke:
  scripts/queue/run_gym_baseline_job.sh

Common options:
  --model MODEL                     HF id or served model name (default: Qwen/Qwen3.5-0.8B)
  --benchmark NAME                  Run a named Gym benchmark
  --environment NAME                Run a named Gym environment
  --resources-server NAME           Run a named resources server
  --agent NAME                      Agent name for direct-input rollouts
  --input PATH                      Direct JSONL input; starts Gym servers then uses --no-serve
  --output PATH                     Rollout JSONL output path
  --split NAME                      Dataset split for benchmark/environment runs
  --limit N                         Maximum tasks to run
  --num-repeats N                   Number of rollouts per task
  --concurrency N                   Rollout concurrency (default: 1)
  --temperature FLOAT               Sampling temperature (default: 0)
  --top-p FLOAT                     Sampling top-p
  --model-extra-body MAPPING        Hydra flow mapping forwarded to the vLLM model wrapper
  --max-output-tokens N             Maximum generated tokens
  --cuda-visible-devices IDS        CUDA_VISIBLE_DEVICES value (default: 1)
  --skip-vllm                       Use an already-running endpoint
  --reuse-gym-servers               Use already-running Gym servers for direct-input mode
  --model-url URL                   OpenAI-compatible /v1 URL (default: http://127.0.0.1:8000/v1)
  --model-api-key KEY               API key sent to the model wrapper (default: EMPTY)
  --vllm-port PORT                  vLLM port (default: 8000)
  --vllm-bin PATH                   vLLM executable (default: vllm)
  --gym-bin PATH                    Gym executable (default: .venv/bin/gym)
  --max-model-len N                 vLLM --max-model-len (default: 8192)
  --gpu-memory-utilization FLOAT    vLLM GPU memory fraction (default: 0.45)
  --tensor-parallel-size N          vLLM tensor parallel size (default: 1)
  --pipeline-parallel-size N        vLLM pipeline parallel size (default: 1)
  --served-model-name NAME          vLLM served model alias
  --vllm-extra-arg ARG              Extra vLLM argv token; repeat as needed
  --gym-start-extra-arg ARG         Extra gym env start argv token; repeat as needed
  --gym-extra-arg ARG               Extra gym eval argv token; repeat as needed
  --config PATH                     Extra Gym config path; repeat as needed
  --config-only                     Use --config paths as the Gym target without --benchmark/--environment
  --search-dir DIR                  Extra Gym search dir; repeat as needed
  --no-prewarm-gym-venvs            Skip sequential per-server virtualenv setup
  --dry-run                         Print commands without running them
  --help                            Show this help
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local flag="$1"
  local value="${2-}"
  if [[ -z "$value" || "$value" == --* ]]; then
    die "$flag requires a value"
  fi
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

safe_name() {
  printf '%s' "$1" | tr '/:' '__' | tr -cs 'A-Za-z0-9_.-' '_'
}

append_if_set() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "$value" ]]; then
    arr_ref+=("$flag" "$value")
  fi
}

prewarm_python_env() {
  local label="$1"
  local dir="$2"

  [[ -d "$dir" ]] || return 0

  if [[ -f "$dir/pyproject.toml" ]]; then
    echo "Prewarming Gym server env ($label): $dir"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      (cd "$dir" && uv sync)
    fi
  elif [[ -f "$dir/requirements.txt" ]]; then
    echo "Prewarming Gym server env ($label): $dir"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      (
        cd "$dir"
        if [[ ! -x .venv/bin/python ]]; then
          uv venv --seed .venv
        fi
        uv pip install --python .venv/bin/python -r requirements.txt
      )
    fi
  fi
}

prewarm_gym_server_envs() {
  [[ "$PREWARM_GYM_VENVS" -eq 1 ]] || return 0

  local resource_dir=""
  local agent_dir=""
  local model_dir=""

  if [[ -n "$RESOURCES_SERVER" ]]; then
    resource_dir="resources_servers/$RESOURCES_SERVER"
  elif [[ -n "$ENVIRONMENT" ]]; then
    resource_dir="resources_servers/$ENVIRONMENT"
  fi

  if [[ -n "$AGENT" ]]; then
    agent_dir="responses_api_agents/$AGENT"
    if [[ ! -d "$agent_dir" && "$AGENT" == *"_simple_agent" ]]; then
      agent_dir="responses_api_agents/simple_agent"
    fi
  fi

  model_dir="responses_api_models/$MODEL_TYPE"

  prewarm_python_env "resources" "$resource_dir"
  prewarm_python_env "agent" "$agent_dir"
  prewarm_python_env "model" "$model_dir"
}

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
MODEL_TYPE="${MODEL_TYPE:-vllm_model}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:8000/v1}"
MODEL_API_KEY="${MODEL_API_KEY:-EMPTY}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-vllm}"
GYM_BIN="${GYM_BIN:-.venv/bin/gym}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
CONCURRENCY="${CONCURRENCY:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-}"
MODEL_EXTRA_BODY="${MODEL_EXTRA_BODY:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
LIMIT="${LIMIT:-}"
NUM_REPEATS="${NUM_REPEATS:-}"
SPLIT="${SPLIT:-}"
BENCHMARK=""
ENVIRONMENT=""
RESOURCES_SERVER=""
AGENT=""
INPUT=""
CONFIG_ONLY=0
OUTPUT=""
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/gym_baseline/${RUN_ID}}"
HEAD_PORT="${HEAD_PORT:-11000}"
GYM_WAIT_TIMEOUT="${GYM_WAIT_TIMEOUT:-180}"
VLLM_WAIT_TIMEOUT="${VLLM_WAIT_TIMEOUT:-600}"
SKIP_VLLM=0
REUSE_GYM_SERVERS=0
DRY_RUN=0
PREWARM_GYM_VENVS="${PREWARM_GYM_VENVS:-1}"
CONFIG_ARGS=()
SEARCH_ARGS=()
VLLM_EXTRA_ARGS=()
GYM_START_EXTRA_ARGS=()
GYM_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) require_value "$1" "${2-}"; MODEL="$2"; shift 2 ;;
    --model-type) require_value "$1" "${2-}"; MODEL_TYPE="$2"; shift 2 ;;
    --model-url) require_value "$1" "${2-}"; MODEL_URL="$2"; shift 2 ;;
    --model-api-key) require_value "$1" "${2-}"; MODEL_API_KEY="$2"; shift 2 ;;
    --benchmark) require_value "$1" "${2-}"; BENCHMARK="$2"; shift 2 ;;
    --environment) require_value "$1" "${2-}"; ENVIRONMENT="$2"; shift 2 ;;
    --resources-server) require_value "$1" "${2-}"; RESOURCES_SERVER="$2"; shift 2 ;;
    --agent) require_value "$1" "${2-}"; AGENT="$2"; shift 2 ;;
    --input) require_value "$1" "${2-}"; INPUT="$2"; shift 2 ;;
    --output) require_value "$1" "${2-}"; OUTPUT="$2"; shift 2 ;;
    --split) require_value "$1" "${2-}"; SPLIT="$2"; shift 2 ;;
    --limit) require_value "$1" "${2-}"; LIMIT="$2"; shift 2 ;;
    --num-repeats) require_value "$1" "${2-}"; NUM_REPEATS="$2"; shift 2 ;;
    --concurrency) require_value "$1" "${2-}"; CONCURRENCY="$2"; shift 2 ;;
    --temperature) require_value "$1" "${2-}"; TEMPERATURE="$2"; shift 2 ;;
    --top-p) require_value "$1" "${2-}"; TOP_P="$2"; shift 2 ;;
    --model-extra-body) require_value "$1" "${2-}"; MODEL_EXTRA_BODY="$2"; shift 2 ;;
    --max-output-tokens) require_value "$1" "${2-}"; MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
    --cuda-visible-devices) require_value "$1" "${2-}"; CUDA_VISIBLE_DEVICES_VALUE="$2"; shift 2 ;;
    --vllm-host) require_value "$1" "${2-}"; VLLM_HOST="$2"; shift 2 ;;
    --vllm-port) require_value "$1" "${2-}"; VLLM_PORT="$2"; MODEL_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"; shift 2 ;;
    --vllm-bin) require_value "$1" "${2-}"; VLLM_BIN="$2"; shift 2 ;;
    --gym-bin) require_value "$1" "${2-}"; GYM_BIN="$2"; shift 2 ;;
    --max-model-len) require_value "$1" "${2-}"; MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) require_value "$1" "${2-}"; GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --tensor-parallel-size) require_value "$1" "${2-}"; TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
    --pipeline-parallel-size) require_value "$1" "${2-}"; PIPELINE_PARALLEL_SIZE="$2"; shift 2 ;;
    --served-model-name) require_value "$1" "${2-}"; SERVED_MODEL_NAME="$2"; shift 2 ;;
    --vllm-wait-timeout) require_value "$1" "${2-}"; VLLM_WAIT_TIMEOUT="$2"; shift 2 ;;
    --gym-wait-timeout) require_value "$1" "${2-}"; GYM_WAIT_TIMEOUT="$2"; shift 2 ;;
    --head-port) require_value "$1" "${2-}"; HEAD_PORT="$2"; shift 2 ;;
    --log-dir) require_value "$1" "${2-}"; LOG_DIR="$2"; shift 2 ;;
    --run-id) require_value "$1" "${2-}"; RUN_ID="$2"; shift 2 ;;
    --config) require_value "$1" "${2-}"; CONFIG_ARGS+=("--config" "$2"); shift 2 ;;
    --config-only) CONFIG_ONLY=1; shift ;;
    --search-dir) require_value "$1" "${2-}"; SEARCH_ARGS+=("--search-dir" "$2"); shift 2 ;;
    --no-prewarm-gym-venvs) PREWARM_GYM_VENVS=0; shift ;;
    --vllm-extra-arg)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      VLLM_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --gym-start-extra-arg)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      GYM_START_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --gym-extra-arg)
      [[ $# -ge 2 ]] || die "$1 requires a value"
      GYM_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --skip-vllm) SKIP_VLLM=1; shift ;;
    --reuse-gym-servers) REUSE_GYM_SERVERS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-gym-${RUN_ID}}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$UV_CACHE_DIR"
fi

# pyproject.toml writes editable-build egg metadata under ./cache. Cryri copies
# intentionally exclude cache contents, so recreate the directory before uv sync.
mkdir -p cache

TARGET_COUNT=0
[[ -n "$BENCHMARK" ]] && TARGET_COUNT=$((TARGET_COUNT + 1))
[[ -n "$ENVIRONMENT" ]] && TARGET_COUNT=$((TARGET_COUNT + 1))
[[ -n "$RESOURCES_SERVER" ]] && TARGET_COUNT=$((TARGET_COUNT + 1))

if [[ "$TARGET_COUNT" -eq 0 && "${#CONFIG_ARGS[@]}" -gt 0 ]]; then
  CONFIG_ONLY=1
  TARGET_COUNT=1
elif [[ "$TARGET_COUNT" -eq 0 && -z "$INPUT" ]]; then
  RESOURCES_SERVER="mcqa"
  AGENT="mcqa_simple_agent"
  INPUT="resources_servers/mcqa/data/example.jsonl"
  LIMIT="${LIMIT:-1}"
  CONCURRENCY="${CONCURRENCY:-1}"
  TARGET_COUNT=1
fi

if [[ "$TARGET_COUNT" -ne 1 ]]; then
  die "choose exactly one of --benchmark, --environment, --resources-server, or --config-only"
fi

if [[ -n "$INPUT" && -z "$AGENT" ]]; then
  if [[ "$RESOURCES_SERVER" == "mcqa" ]]; then
    AGENT="mcqa_simple_agent"
  else
    die "--input mode requires --agent"
  fi
fi

if [[ -z "$OUTPUT" ]]; then
  TARGET_NAME="${BENCHMARK:-${ENVIRONMENT:-${RESOURCES_SERVER}}}"
  OUTPUT="results/baselines/$(safe_name "$MODEL")/$(safe_name "$TARGET_NAME")_${RUN_ID}.jsonl"
fi

TARGET_ARGS=()
if [[ -n "$BENCHMARK" ]]; then
  TARGET_ARGS+=("--benchmark" "$BENCHMARK")
elif [[ -n "$ENVIRONMENT" ]]; then
  TARGET_ARGS+=("--environment" "$ENVIRONMENT")
elif [[ -n "$RESOURCES_SERVER" ]]; then
  TARGET_ARGS+=("--resources-server" "$RESOURCES_SERVER")
fi

VLLM_PID=""
GYM_PID=""
CLEANUP_DONE=0

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return 0
  fi
  CLEANUP_DONE=1

  if [[ -n "${GYM_PID:-}" ]] && kill -0 "$GYM_PID" 2>/dev/null; then
    echo "Stopping Gym servers (PID $GYM_PID)"
    kill "$GYM_PID" 2>/dev/null || true
    wait "$GYM_PID" 2>/dev/null || true
  fi
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM (PID $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}

handle_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup
  exit "$rc"
}

handle_signal() {
  local rc="$1"
  trap - EXIT INT TERM
  cleanup
  exit "$rc"
}

trap handle_exit EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if [[ ! -x "$GYM_BIN" ]]; then
  if command -v gym >/dev/null 2>&1; then
    GYM_BIN="$(command -v gym)"
  else
    die "Gym executable '$GYM_BIN' not found; run from the repo root or pass --gym-bin"
  fi
fi

if [[ "$SKIP_VLLM" -eq 0 && "$DRY_RUN" -eq 0 ]] && ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  die "vLLM executable '$VLLM_BIN' not found. Install the optional extra with a local cache, e.g. UV_CACHE_DIR=/tmp/uv-cache-gym-vllm TMPDIR=/tmp uv pip install --python /tmp/gym-vllm-venv/bin/python '.[vllm]', then pass --vllm-bin /tmp/gym-vllm-venv/bin/vllm; or pass --skip-vllm --model-url <running-endpoint>/v1."
elif [[ "$SKIP_VLLM" -eq 0 && "$DRY_RUN" -eq 1 ]] && ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "Warning: vLLM executable '$VLLM_BIN' not found; dry-run will still print commands." >&2
fi

mkdir -p "$(dirname "$OUTPUT")" "$LOG_DIR"

echo "Run id: $RUN_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES_VALUE"
echo "Model: $MODEL"
echo "Model URL: $MODEL_URL"
echo "Output: $OUTPUT"
echo "Logs: $LOG_DIR"
echo "UV cache: $UV_CACHE_DIR"

prewarm_gym_server_envs

if [[ "$SKIP_VLLM" -eq 0 ]]; then
  VLLM_CMD=(
    "$VLLM_BIN" serve "$MODEL"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
  )
  if [[ -n "$SERVED_MODEL_NAME" ]]; then
    VLLM_CMD+=(--served-model-name "$SERVED_MODEL_NAME")
  fi
  if [[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64:-}" ]]; then
    VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="$(printf '%s' "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64" | base64 -d)"
  fi
  if [[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]]; then
    VLLM_CMD+=(--default-chat-template-kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS")
  fi
  VLLM_CMD+=("${VLLM_EXTRA_ARGS[@]}")

  echo "vLLM command:"
  quote_cmd CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${VLLM_CMD[@]}"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${VLLM_CMD[@]}" >"$LOG_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
  fi
else
  echo "Skipping vLLM start; using existing endpoint."
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "Waiting for model endpoint..."
  for _ in $(seq 1 "$VLLM_WAIT_TIMEOUT"); do
    if curl -sf "${MODEL_URL%/}/models" >/dev/null 2>&1; then
      echo "Model endpoint is ready"
      break
    fi
    if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
      tail -n 80 "$LOG_DIR/vllm.log" >&2 || true
      die "vLLM exited before becoming ready"
    fi
    sleep 1
  done
  if ! curl -sf "${MODEL_URL%/}/models" >/dev/null 2>&1; then
    [[ -f "$LOG_DIR/vllm.log" ]] && tail -n 80 "$LOG_DIR/vllm.log" >&2 || true
    die "model endpoint did not become ready within ${VLLM_WAIT_TIMEOUT}s"
  fi
fi

COMMON_EVAL_ARGS=(
  --output "$OUTPUT"
  --concurrency "$CONCURRENCY"
  --temperature "$TEMPERATURE"
)
append_if_set COMMON_EVAL_ARGS --limit "$LIMIT"
append_if_set COMMON_EVAL_ARGS --num-repeats "$NUM_REPEATS"
append_if_set COMMON_EVAL_ARGS --top-p "$TOP_P"
append_if_set COMMON_EVAL_ARGS --max-output-tokens "$MAX_OUTPUT_TOKENS"
COMMON_EVAL_ARGS+=("${GYM_EXTRA_ARGS[@]}")

MODEL_EXTRA_BODY_ARG=()
if [[ -n "$MODEL_EXTRA_BODY" ]]; then
  MODEL_EXTRA_BODY_ARG=("++policy_model.responses_api_models.vllm_model.extra_body=${MODEL_EXTRA_BODY}")
fi

if [[ -n "$INPUT" ]]; then
  if [[ "$REUSE_GYM_SERVERS" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    if curl -sf "http://127.0.0.1:${HEAD_PORT}/server_instances" >/dev/null 2>&1; then
      die "Gym head server is already reachable on port ${HEAD_PORT}; stop it or pass --reuse-gym-servers"
    fi
  fi

  if [[ "$REUSE_GYM_SERVERS" -eq 0 ]]; then
    GYM_START_CMD=(
      "$GYM_BIN" env start
      "${CONFIG_ARGS[@]}"
      "${TARGET_ARGS[@]}"
      --model-type "$MODEL_TYPE"
      "${SEARCH_ARGS[@]}"
      --model "$MODEL"
      --model-url "$MODEL_URL"
      --model-api-key "$MODEL_API_KEY"
      "${MODEL_EXTRA_BODY_ARG[@]}"
      "${GYM_START_EXTRA_ARGS[@]}"
    )
    echo "Gym server command:"
    quote_cmd "${GYM_START_CMD[@]}"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      "${GYM_START_CMD[@]}" >"$LOG_DIR/gym_servers.log" 2>&1 &
      GYM_PID=$!
      scripts/wait_for_servers.sh "$GYM_PID" "$HEAD_PORT" "$GYM_WAIT_TIMEOUT" | tee "$LOG_DIR/wait_for_servers.log"
    fi
  else
    echo "Reusing already-running Gym servers."
  fi

  EVAL_CMD=(
    "$GYM_BIN" eval run
    --no-serve
    --agent "$AGENT"
    --input "$INPUT"
    "${COMMON_EVAL_ARGS[@]}"
  )
else
  EVAL_CMD=(
    "$GYM_BIN" eval run
    "${CONFIG_ARGS[@]}"
    "${TARGET_ARGS[@]}"
    --model-type "$MODEL_TYPE"
    "${SEARCH_ARGS[@]}"
    --model "$MODEL"
    --model-url "$MODEL_URL"
    --model-api-key "$MODEL_API_KEY"
    "${MODEL_EXTRA_BODY_ARG[@]}"
    "${COMMON_EVAL_ARGS[@]}"
  )
  append_if_set EVAL_CMD --split "$SPLIT"
  append_if_set EVAL_CMD --agent "$AGENT"
fi

echo "Gym eval command:"
quote_cmd "${EVAL_CMD[@]}"
if [[ "$DRY_RUN" -eq 0 ]]; then
  "${EVAL_CMD[@]}" 2>&1 | tee "$LOG_DIR/gym_eval.log"
fi

echo "Done"
