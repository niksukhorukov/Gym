# Decomposer agent

## Workplace Assistant

The GLM-5.2 Workplace config uses GLM-5.2 through OpenRouter as the Decomposer
and thinking and non-thinking variants of Gemma-4-E2B, Gemma-4-E4B,
Gemma-4-12B, and Gemma-4-26B-A4B as eight subagent types. Set `policy_api_key`
in `external/Gym/env.yaml` to an OpenRouter API key. From the project root,
serve the four subagent models, each in its own terminal:

```bash
scripts/vllm_serve_gemma_4_e2b.sh
scripts/vllm_serve_gemma_4_e4b.sh
scripts/vllm_serve_gemma_4_12b.sh
scripts/vllm_serve_gemma_4_26b_a4b.sh
```

Start `subagent_server/serve.sh`, then start Gym with:

```bash
cd external/Gym
source .venv/bin/activate

gym env start \
  --config responses_api_agents/decomposer_agent/configs/workplace_assistant_glm_5_2.yaml
```

Run one task:

```bash
gym eval run --no-serve \
  --agent workplace_assistant_decomposer_agent \
  --input resources_servers/workplace_assistant/data/train.jsonl \
  --output ../../artifacts/workplace_assistant_glm_5_2_smoke.jsonl \
  --limit 1 \
  --num-repeats 1 \
  --concurrency 1
```
