# Decomposer subagent server

This lightweight LangGraph server exposes the eight assistants registered in
`langgraph.json`. Each assistant is compiled in `subagents.py` as a LangChain
agent that receives one delegated task and returns a self-contained report.

Tools are supplied per run through LangGraph runtime context. The shared
`NeMoGymSubagentMiddleware` converts Gym Responses API function schemas to Chat
Completions schemas, exposes them to the model, and executes generated calls by
posting their arguments to the seeded Gym resource server. Runs without Gym
function tools retain the previous tool-free behavior.

| Assistant ID | Model | Endpoint | Thinking |
| --- | --- | --- | --- |
| `gemma_4_e2b_thinking` | `google/gemma-4-E2B-it` | `http://127.0.0.1:8020/v1` | Enabled |
| `gemma_4_e2b_non_thinking` | `google/gemma-4-E2B-it` | `http://127.0.0.1:8020/v1` | Disabled |
| `gemma_4_e4b_thinking` | `google/gemma-4-E4B-it` | `http://127.0.0.1:8021/v1` | Enabled |
| `gemma_4_e4b_non_thinking` | `google/gemma-4-E4B-it` | `http://127.0.0.1:8021/v1` | Disabled |
| `gemma_4_12b_thinking` | `google/gemma-4-12B-it` | `http://127.0.0.1:8022/v1` | Enabled |
| `gemma_4_12b_non_thinking` | `google/gemma-4-12B-it` | `http://127.0.0.1:8022/v1` | Disabled |
| `gemma_4_26b_a4b_thinking` | `google/gemma-4-26B-A4B-it` | `http://127.0.0.1:8023/v1` | Enabled |
| `gemma_4_26b_a4b_non_thinking` | `google/gemma-4-26B-A4B-it` | `http://127.0.0.1:8023/v1` | Disabled |

Thinking assistants request and preserve reasoning output. Where supported,
non-thinking assistants explicitly disable it. No explicit thinking-token
budget is set.

The assistants do not set an explicit completion limit, so vLLM uses the model
context remaining after the prompt. All eight use a 300-second request timeout,
no retries, disabled streaming, and the Chat Completions API. Their sampling
parameters are:

| Model mode | Temperature | `top_p` | `top_k` | Other parameters |
| --- | ---: | ---: | ---: | --- |
| Gemma thinking and non-thinking | 1.0 | 0.95 | 64 | — |

Start the required local vLLM servers, then run:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagent_server/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. `serve.sh` launches
`langgraph dev` using `langgraph.json` in an isolated uv environment, so its
dependencies do not alter the root or Gym environments. `HOST` and `PORT` can
override the default bind address and port.
