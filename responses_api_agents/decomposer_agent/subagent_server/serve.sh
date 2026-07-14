#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run \
  --isolated \
  --no-project \
  --with-requirements requirements.txt \
  langgraph dev \
  --config langgraph.json \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-2024}" \
  --no-browser \
  --no-reload
