# Decomposer subagent server

This lightweight LangGraph server exposes the three local model-backed
assistants used by `DecomposerAgent`:

| Assistant ID | Default model | Default endpoint |
| --- | --- | --- |
| `qwen3_5_4b` | `Qwen/Qwen3.5-4B` | `http://127.0.0.1:8012/v1` |
| `gemma_4_e4b` | `google/gemma-4-E4B-it` | `http://127.0.0.1:8013/v1` |
| `lfm2_5_8b_a1b` | `LiquidAI/LFM2.5-8B-A1B` | `http://127.0.0.1:8014/v1` |

Each assistant is a zero-tool LangChain agent. It receives one delegated task
and returns a self-contained report. Start all three vLLM servers first, then
run:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagent_server/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. Override `HOST` or
`PORT` for the server itself.

The launcher uses an isolated uv environment, so `langchain-openai` and the
LangGraph development server do not alter either the root or Gym environment.
