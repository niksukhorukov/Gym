from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from chat_vllm import ChatVLLM


SUBAGENT_SYSTEM_PROMPT = """You are a helpful assistant.

Given a task, you work until the task is done or cannot be completed for some reason. Then, you always respond to the user with an honest and self-contained final report.
"""

THINKING_TOKEN_BUDGET = 4096
MAX_COMPLETION_TOKENS = 5120
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 0


def qwen3_5_4b() -> CompiledStateGraph:
    model = ChatVLLM(
        model="Qwen/Qwen3.5-4B",
        base_url="http://127.0.0.1:8012/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        presence_penalty=1.5,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        preserve_tool_call_reasoning=True,
        extra_body={
            "top_k": 20,
            "thinking_token_budget": THINKING_TOKEN_BUDGET,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)


def gemma_4_e4b() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E4B-it",
        base_url="http://127.0.0.1:8013/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        preserve_tool_call_reasoning=True,
        extra_body={
            "top_k": 64,
            "thinking_token_budget": THINKING_TOKEN_BUDGET,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)


def lfm2_5_8b_a1b() -> CompiledStateGraph:
    model = ChatVLLM(
        model="LiquidAI/LFM2.5-8B-A1B",
        base_url="http://127.0.0.1:8014/v1",
        api_key="EMPTY",
        temperature=0.2,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 80,
            "repetition_penalty": 1.05,
            "thinking_token_budget": THINKING_TOKEN_BUDGET,
            "include_reasoning": False,
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SUBAGENT_SYSTEM_PROMPT)
