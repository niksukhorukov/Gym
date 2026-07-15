#!/usr/bin/env bash
# Build a Cryri submission for scripts/queue/run_gym_baseline_job.sh.
#
# This script is dry-run by default. It prints the generated YAML and only
# invokes cryri when --submit is passed explicitly.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'USAGE'
Usage:
  scripts/queue/submit_gym_baseline_cryri.sh [wrapper options] [runner options]

Wrapper options:
  --submit                    Submit with cryri. Default is dry-run only.
  --dry-run                   Generate and print YAML without submitting.
  --image IMAGE               Container image. Also read from DOCKER_IMAGE.
  --copy-dir PATH             Cryri copy dir. Also read from COPY_DIR; default is SR006.nfs3.
  --copy-exclude PATTERN      Add to the default Cryri copy exclusions (repeatable).
  --region REGION             Cryri region (default: SR006 or REGION env)
  --instance-type TYPE        Cryri instance type
  --priority PRIORITY         Cryri priority (default: medium)
  --description TEXT          Cryri job description
  --yaml-output PATH          Write YAML to this path instead of /tmp
  --config-env PATH           Optional env file to source (default: config/cloud.env)
  --help                      Show this help

All other options are forwarded to run_gym_baseline_job.sh.
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

yaml_sq() {
  local value="$1"
  value="${value//\'/\'\'}"
  printf "'%s'" "$value"
}

quote_cmd() {
  printf '%q ' "$@"
}

ORIGINAL_ARGS=("$@")
CONFIG_ENV="${CONFIG_ENV:-config/cloud.env}"
CONFIG_ENV_EXPLICIT=0

# Help must not depend on the presence or contents of a configuration file.
for arg in "${ORIGINAL_ARGS[@]}"; do
  if [[ "$arg" == "--help" ]]; then
    usage
    exit 0
  fi
done

# Find the config before parsing the remaining options so it can provide
# defaults. Only the last --config-env is sourced, matching the full parser's
# last-option-wins behavior.
for ((arg_index = 0; arg_index < ${#ORIGINAL_ARGS[@]}; arg_index++)); do
  case "${ORIGINAL_ARGS[$arg_index]}" in
    --config-env)
      value_index=$((arg_index + 1))
      config_value="${ORIGINAL_ARGS[$value_index]-}"
      require_value "--config-env" "$config_value"
      CONFIG_ENV="$config_value"
      CONFIG_ENV_EXPLICIT=1
      arg_index="$value_index"
      ;;
  esac
done

SELECTED_CONFIG_ENV="$CONFIG_ENV"
if [[ -f "$SELECTED_CONFIG_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$SELECTED_CONFIG_ENV"
  set +a
elif [[ "$CONFIG_ENV_EXPLICIT" -eq 1 ]]; then
  die "config env file does not exist: $SELECTED_CONFIG_ENV"
fi

# A sourced shell file can change positional parameters, the working
# directory, or shell options. Restore the wrapper's execution context before
# applying defaults and parsing the original command line.
set -euo pipefail
set -- "${ORIGINAL_ARGS[@]}"
cd "$REPO_ROOT"
CONFIG_ENV="$SELECTED_CONFIG_ENV"

SUBMIT=0
DOCKER_IMAGE="${DOCKER_IMAGE:-${IMAGE:-}}"
COPY_DIR="${COPY_DIR:-${CRY_COPY_DIR:-}}"
REGION="${REGION:-SR006}"
INSTANCE_TYPE="${INSTANCE_TYPE:-a100plus.1gpu.80vG.12C.96G}"
PRIORITY="${PRIORITY:-medium}"
DESCRIPTION="${DESCRIPTION:-Gym local-model baseline}"
YAML_OUTPUT="${YAML_OUTPUT:-}"
RUNNER_ARGS=()
COPY_EXCLUDES=()

# Capture the job environment only after loading configuration defaults.
DEFAULT_HF_HOME="/home/jovyan/shares/SR006.nfs3/.cache/huggingface"
JOB_HF_HOME="${HF_HOME:-$DEFAULT_HF_HOME}"
JOB_HF_HUB_CACHE="${HF_HUB_CACHE:-$JOB_HF_HOME/hub}"
JOB_HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
JOB_VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
JOB_VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-}"
JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64="${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64:-}"
JOB_PYTHONPATH="${PYTHONPATH:-.:/workspace/Gym}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --dry-run) SUBMIT=0; shift ;;
    --image) require_value "$1" "${2-}"; DOCKER_IMAGE="$2"; shift 2 ;;
    --copy-dir) require_value "$1" "${2-}"; COPY_DIR="$2"; shift 2 ;;
    --copy-exclude) require_value "$1" "${2-}"; COPY_EXCLUDES+=("$2"); shift 2 ;;
    --region) require_value "$1" "${2-}"; REGION="$2"; shift 2 ;;
    --instance-type) require_value "$1" "${2-}"; INSTANCE_TYPE="$2"; shift 2 ;;
    --priority) require_value "$1" "${2-}"; PRIORITY="$2"; shift 2 ;;
    --description) require_value "$1" "${2-}"; DESCRIPTION="$2"; shift 2 ;;
    --yaml-output) require_value "$1" "${2-}"; YAML_OUTPUT="$2"; shift 2 ;;
    --config-env) require_value "$1" "${2-}"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) RUNNER_ARGS+=("$1"); shift ;;
  esac
done

COPY_DIR="${COPY_DIR:-/home/jovyan/shares/SR006.nfs3/sukhorukov/cryri_copies/gym-baseline-${USER:-user}}"

if [[ -z "$DOCKER_IMAGE" ]]; then
  if [[ "$SUBMIT" -eq 1 ]]; then
    die "DOCKER_IMAGE is required for --submit; set it, pass --image, or create config/cloud.env"
  fi
  DOCKER_IMAGE="MISSING_DOCKER_IMAGE"
  echo "Warning: DOCKER_IMAGE is not set; dry-run YAML uses MISSING_DOCKER_IMAGE" >&2
fi

if [[ -z "$YAML_OUTPUT" ]]; then
  YAML_OUTPUT="$(mktemp /tmp/gym-baseline-cryri.XXXXXX.yaml)"
fi

if [[ "$SUBMIT" -eq 1 ]]; then
  mkdir -p "$COPY_DIR"
fi

RUNNER_SCRIPT="scripts/queue/run_gym_baseline_job.sh"
COMMAND="$(quote_cmd bash "$RUNNER_SCRIPT" "${RUNNER_ARGS[@]}")"

DEFAULT_COPY_EXCLUDES=(
  ".git"
  ".venv"
  "__pycache__"
  ".pytest_cache"
  ".ruff_cache"
  ".mypy_cache"
  ".cache"
  "cache"
  "shares"
  "env.yaml"
  ".env"
  ".env.*"
  "logs"
  "outputs"
  "results"
  "resources_servers/mcqa/data/train.jsonl"
  "resources_servers/*/.venv"
  "responses_api_agents/*/.venv"
  "responses_api_models/*/.venv"
  "environments/*/.venv"
  "resources_servers/*/data/train.jsonl"
  "*.pyc"
)

{
  echo "container:"
  echo "  image: $(yaml_sq "$DOCKER_IMAGE")"
  echo "  command: $(yaml_sq "$COMMAND")"
  echo "  environment:"
  echo "    HF_HOME: $(yaml_sq "$JOB_HF_HOME")"
  echo "    HF_HUB_CACHE: $(yaml_sq "$JOB_HF_HUB_CACHE")"
  echo "    HF_XET_HIGH_PERFORMANCE: $(yaml_sq "$JOB_HF_XET_HIGH_PERFORMANCE")"
  echo "    VLLM_USE_FLASHINFER_SAMPLER: $(yaml_sq "$JOB_VLLM_USE_FLASHINFER_SAMPLER")"
  if [[ -n "$JOB_VLLM_USE_V2_MODEL_RUNNER" ]]; then
    echo "    VLLM_USE_V2_MODEL_RUNNER: $(yaml_sq "$JOB_VLLM_USE_V2_MODEL_RUNNER")"
  fi
  if [[ -n "$JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS" ]]; then
    echo "    VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS: $(yaml_sq "$JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS")"
  fi
  if [[ -n "$JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64" ]]; then
    echo "    VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64: $(yaml_sq "$JOB_VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS_B64")"
  fi
  echo "    PYTHONPATH: $(yaml_sq "$JOB_PYTHONPATH")"
  echo "  work_dir: '.'"
  echo "  run_from_copy: True"
  echo "  cry_copy_dir: $(yaml_sq "$COPY_DIR")"
  echo "  exclude_from_copy:"
  for pattern in "${DEFAULT_COPY_EXCLUDES[@]}" "${COPY_EXCLUDES[@]}"; do
    echo "    - $(yaml_sq "$pattern")"
  done
  echo "cloud:"
  echo "  region: $(yaml_sq "$REGION")"
  echo "  instance_type: $(yaml_sq "$INSTANCE_TYPE")"
  echo "  n_workers: 1"
  echo "  priority: $(yaml_sq "$PRIORITY")"
  echo "  description: $(yaml_sq "$DESCRIPTION")"
} >"$YAML_OUTPUT"

echo "Generated Cryri YAML: $YAML_OUTPUT"
echo "Runner command:"
echo "$COMMAND"
echo ""
cat "$YAML_OUTPUT"

if [[ "$SUBMIT" -eq 1 ]]; then
  echo "Submitting with cryri..."
  cryri "$YAML_OUTPUT"
else
  echo "Dry run only. Re-run with --submit to submit."
fi
