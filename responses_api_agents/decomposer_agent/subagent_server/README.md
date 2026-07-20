# Decomposer subagent server

This lightweight LangGraph server exposes the four assistants registered in
`langgraph.json`. Each assistant is compiled in `graph.py` as a zero-tool
LangChain agent that receives one delegated task and returns a self-contained
report.

| Assistant ID | Model | Endpoint | Thinking |
| --- | --- | --- | --- |
| `qwen3_6_35b_a3b_fp8_thinking` | `Qwen/Qwen3.6-35B-A3B-FP8` | `http://127.0.0.1:8019/v1` | Enabled |
| `qwen3_6_35b_a3b_fp8_non_thinking` | `Qwen/Qwen3.6-35B-A3B-FP8` | `http://127.0.0.1:8019/v1` | Disabled |
| `gemma_4_26b_a4b_thinking` | `google/gemma-4-26B-A4B-it` | `http://127.0.0.1:8020/v1` | Enabled |
| `gemma_4_26b_a4b_non_thinking` | `google/gemma-4-26B-A4B-it` | `http://127.0.0.1:8020/v1` | Disabled |

Thinking assistants request and preserve reasoning output. Non-thinking
assistants explicitly disable it. No explicit thinking-token budget is set.

All four assistants use a 32,768-token completion limit, a 300-second request
timeout, no retries, one completion, disabled streaming, and the Chat
Completions API. Their sampling parameters are:

| Model mode | Temperature | `top_p` | `top_k` | Other parameters |
| --- | ---: | ---: | ---: | --- |
| Qwen thinking | 1.0 | 0.95 | 20 | `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` |
| Qwen non-thinking | 0.7 | 0.8 | 20 | `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0` |
| Gemma thinking and non-thinking | 1.0 | 0.95 | 64 | — |

`graph.py` also defines `gemma_4_e2b_thinking` and
`gemma_4_e2b_non_thinking`, but they are not registered in `langgraph.json`
and therefore are not exposed by this server.

Start the required local vLLM servers, then run:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagent_server/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. `serve.sh` launches
`langgraph dev` using `langgraph.json` in an isolated uv environment, so its
dependencies do not alter the root or Gym environments. `HOST` and `PORT` can
override the default bind address and port.
