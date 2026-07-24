# Decomposer agent

## Workplace Assistant

The Workplace config uses GLM-5.2 through OpenRouter as the Decomposer,
thinking and non-thinking variants of Gemma-4-E4B and Qwen3.5-4B, and the
thinking LFM2.5-8B-A1B as subagents. Set `policy_api_key` in
`external/Gym/env.yaml` to an OpenRouter API key. From the project root, serve
the three subagent models, each in its own terminal:

```bash
scripts/vllm_serve_gemma_4_e4b.sh
scripts/vllm_serve_qwen3_5_4b.sh
scripts/vllm_serve_lfm2_5_8b_a1b.sh
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
