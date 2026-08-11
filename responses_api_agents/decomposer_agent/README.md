# Decomposer agent

## Workplace Assistant

The Workplace config uses DeepSeek V4 Flash 0731 through OpenRouter as the
Decomposer and Gemma-4-26B-A4B with thinking disabled as the subagent type. Set
`policy_api_key` in `external/Gym/env.yaml` to an OpenRouter API key. From the
project root, serve the subagent model:

```bash
scripts/vllm/serve_gemma_4_26b_a4b.sh
```

Start the subagent server:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagents/serve.sh
```

Then start Gym with:

```bash
cd external/Gym
source .venv/bin/activate

gym env start \
  --config responses_api_agents/decomposer_agent/configs/workplace_assistant_deepseek_v4_flash_20260731.yaml
```

Run one task:

```bash
gym eval run --no-serve \
  --agent workplace_assistant_decomposer_agent \
  --input resources_servers/workplace_assistant/data/train.jsonl \
  --output ../../artifacts/workplace_assistant_deepseek_v4_flash_20260731_smoke.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```
