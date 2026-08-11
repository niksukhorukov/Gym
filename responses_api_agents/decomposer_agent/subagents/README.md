# Decomposer subagent server

This lightweight LangGraph server exposes the
`gemma_4_26b_a4b_non_thinking` assistant registered in `langgraph.json`. It is
compiled in `graph.py` as a LangChain agent that receives one delegated task
and returns a self-contained report.

Tools are supplied per run through LangGraph runtime context. The shared
`NeMoGymSubagentMiddleware` converts Gym Responses API function schemas to Chat
Completions schemas, exposes them to the model, and executes generated calls by
posting their arguments to the seeded Gym resource server. Runs without Gym
function tools retain the previous tool-free behavior.

| Assistant ID | Model | Endpoint | Thinking |
| --- | --- | --- | --- |
| `gemma_4_26b_a4b_non_thinking` | `google/gemma-4-26B-A4B-it` | `http://127.0.0.1:8023/v1` | Disabled |

The assistant explicitly disables reasoning output. No explicit completion
limit is set, so vLLM uses the model context remaining after the prompt. It
uses a 300-second request timeout, no retries, disabled streaming, and the Chat
Completions API.

Its sampling parameters are:

| Model mode | Temperature | `top_p` | `top_k` | Other parameters |
| --- | ---: | ---: | ---: | --- |
| Gemma non-thinking | 1.0 | 0.95 | 64 | — |

Start the required local vLLM server, then run:

```bash
external/Gym/responses_api_agents/decomposer_agent/subagents/serve.sh
```

The server listens on `http://127.0.0.1:2024` by default. Its launcher uses this
directory's `requirements.txt` in an isolated uv environment, so its dependencies
do not alter the root or Gym environments. `HOST` and `PORT` can override the
default bind address and port.
