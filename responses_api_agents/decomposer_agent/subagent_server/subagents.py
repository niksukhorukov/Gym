from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from chat_vllm import ChatVLLM


SYSTEM_PROMPT = """You are a helpful general-purpose assistant. Answer briefly and to the point."""

MAX_COMPLETION_TOKENS = 32768
REQUEST_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 0


def gemma_4_e2b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E2B-it",
        base_url="http://127.0.0.1:8016/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def gemma_4_e2b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E2B-it",
        base_url="http://127.0.0.1:8016/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def qwen3_6_35b_a3b_fp8_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        base_url="http://127.0.0.1:8019/v1",
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
        preserve_reasoning=True,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def qwen3_6_35b_a3b_fp8_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="Qwen/Qwen3.6-35B-A3B-FP8",
        base_url="http://127.0.0.1:8019/v1",
        api_key="EMPTY",
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def gemma_4_26b_a4b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8020/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
            "include_reasoning": True,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)


def gemma_4_26b_a4b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8020/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_TOKENS,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        n=1,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return create_agent(model=model, tools=[], system_prompt=SYSTEM_PROMPT)
