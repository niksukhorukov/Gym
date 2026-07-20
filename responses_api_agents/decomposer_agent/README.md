# Decomposer agent

## GPQA Diamond smoke run

This example uses Gemma-4 as the Decomposer and the four Qwen/Gemma assistants
exposed by `subagent_server`.

Before starting, make sure:

- Qwen3.6-35B-A3B-FP8 is served at `http://127.0.0.1:8019/v1`.
- Gemma-4-26B-A4B-IT is served at `http://127.0.0.1:8020/v1`.
- `resources_servers/gpqa_diamond/data/train.jsonl` exists. See the
  [GPQA Diamond README](../../resources_servers/gpqa_diamond/README.md) if it
  needs to be generated.

From the repository root, start the subagent server:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagent_server/serve.sh
```

In a second terminal, start the Gym servers:

```bash
cd external/Gym
source .venv/bin/activate

gym env start \
  --config responses_api_agents/decomposer_agent/configs/gpqa_diamond_gemma_4_26b_a4b.yaml
```

In a third terminal, run one task:

```bash
cd external/Gym
source .venv/bin/activate

gym eval run --no-serve \
  --agent gpqa_diamond_decomposer_agent \
  --input resources_servers/gpqa_diamond/data/train.jsonl \
  --output ../../artifacts/gpqa_diamond_gemma_smoke.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```

Stop the Gym and subagent servers with `Ctrl+C` when the run finishes. Other
GPQA Decomposer models can be selected by replacing the config passed to
`gym env start` with another file from `configs/`.
