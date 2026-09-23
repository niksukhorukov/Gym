# Decomposer subagent server

This lightweight LangGraph server exposes the assistants registered in
`langgraph.json` and compiled in `graph.py`.

Tools are supplied per run through LangGraph runtime context. The shared
`NeMoGymSubagentMiddleware` converts Gym Responses API function schemas to Chat
Completions schemas, exposes them to the model, and executes generated calls by
posting their arguments to the seeded Gym resource server. Runs without Gym
function tools retain the previous tool-free behavior.

| Assistant ID | Model | Endpoint | Thinking |
| --- | --- | --- | --- |
| `qwen_3_5_4b_non_thinking` | `Qwen/Qwen3.5-4B` | `http://127.0.0.1:8024/v1` | Disabled |

Thinking and reasoning output are disabled. No explicit
completion limit is set, so vLLM uses the model context remaining after the
prompt. It uses a 300-second request timeout, no retries, disabled streaming,
and the Chat Completions API.

Its sampling parameters are:

| Model mode | Temperature | `top_p` | `top_k` | Other parameters |
| --- | ---: | ---: | ---: | --- |
| Qwen3.5-4B non-thinking | 0.7 | 0.8 | 20 | `presence_penalty=1.5`, `min_p=0.0`, `repetition_penalty=1.0` |

Start the local vLLM server with `scripts/vllm/serve_qwen_3_5_4b.sh`, then run:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagents/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. Its launcher uses this
directory's `requirements.txt` in an isolated uv environment, so its dependencies
do not alter the root or Gym environments. `HOST` and `PORT` can override the
default bind address and port.
