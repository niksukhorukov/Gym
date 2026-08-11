import json
from collections.abc import Awaitable, Callable
from typing import Any

from decomposer.prompts import SUBAGENT_SYSTEM_PROMPT
from decomposer.chat_vllm import ChatVLLM
from httpx import AsyncClient, RequestError
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.graph.state import CompiledStateGraph


REQUEST_TIMEOUT_SECONDS = 300.0
RESOURCE_SERVER_TIMEOUT_SECONDS = 300.0
MAX_RETRIES = 0


class NeMoGymSubagentMiddleware(AgentMiddleware):
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        context: dict[str, Any] = request.runtime.context
        body = context["body"]
        tools = [
            _to_chat_completions_tool(tool)
            for tool in body.get("tools", [])
        ]
        if not tools:
            return await handler(request.override(tools=[]))

        model_settings = {
            **request.model_settings,
            "parallel_tool_calls": body.get("parallel_tool_calls", True),
        }
        return await handler(
            request.override(
                tools=tools,
                tool_choice=_to_chat_completions_tool_choice(
                    body.get("tool_choice", "auto")
                ),
                model_settings=model_settings,
            )
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        context: dict[str, Any] = request.runtime.context
        tool_call = request.tool_call
        available_tool_names = [
            tool["name"]
            for tool in context["body"].get("tools", [])
            if tool.get("type") == "function" and isinstance(tool.get("name"), str)
        ]
        if tool_call["name"] not in available_tool_names:
            return ToolMessage(
                content=(
                    f"Error: {tool_call['name']} is not a valid tool, "
                    f"try one of [{', '.join(available_tool_names)}]."
                ),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )

        url = f"{context['resource_server_url'].rstrip('/')}/{tool_call['name']}"
        try:
            async with AsyncClient(timeout=RESOURCE_SERVER_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json=tool_call["args"],
                    cookies=context["resource_server_cookies"],
                )
        except RequestError as error:
            return ToolMessage(
                content=json.dumps({"error": f"Gym resource-server request failed: {error}"}),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )

        context["resource_server_cookies"].update(dict(response.cookies))
        return ToolMessage(
            content=response.text,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
            status="error" if response.is_error else "success",
        )


def _to_chat_completions_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") != "function":
        raise ValueError(
            "NeMoGymSubagentMiddleware supports only Gym resource-server function tools; "
            f"received type {tool.get('type')!r}."
        )
    if not isinstance(tool.get("name"), str):
        raise ValueError("Gym resource-server function tools must have a string `name`.")

    return {
        "type": "function",
        "function": {
            key: tool[key]
            for key in ("name", "description", "parameters", "strict")
            if key in tool
        },
    }


def _to_chat_completions_tool_choice(
    tool_choice: str | dict[str, Any],
) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        return tool_choice
    if tool_choice.get("type") != "function":
        raise ValueError(
            "NeMoGymSubagentMiddleware supports only string or function "
            f"Gym tool choices; received type {tool_choice.get('type')!r}."
        )

    name = tool_choice.get("name")
    if not isinstance(name, str):
        raise ValueError("Gym function tool choices must have a string `name`.")
    return {
        "type": "function",
        "function": {"name": name},
    }


def _create_subagent(model: ChatVLLM) -> CompiledStateGraph:
    return create_agent(
        model=model,
        tools=[],
        middleware=[NeMoGymSubagentMiddleware()],
        system_prompt=SUBAGENT_SYSTEM_PROMPT,
    )


def gemma_4_2b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E2B-it",
        base_url="http://127.0.0.1:8020/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
        },
    )
    return _create_subagent(model)


def gemma_4_2b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E2B-it",
        base_url="http://127.0.0.1:8020/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _create_subagent(model)


def gemma_4_4b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E4B-it",
        base_url="http://127.0.0.1:8021/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
        },
    )
    return _create_subagent(model)


def gemma_4_4b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-E4B-it",
        base_url="http://127.0.0.1:8021/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _create_subagent(model)


def gemma_4_12b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-12B-it",
        base_url="http://127.0.0.1:8022/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
        },
    )
    return _create_subagent(model)


def gemma_4_12b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-12B-it",
        base_url="http://127.0.0.1:8022/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _create_subagent(model)


def gemma_4_26b_a4b_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8023/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=True,
        extra_body={
            "top_k": 64,
        },
    )
    return _create_subagent(model)


def gemma_4_26b_a4b_non_thinking() -> CompiledStateGraph:
    model = ChatVLLM(
        model="google/gemma-4-26B-A4B-it",
        base_url="http://127.0.0.1:8023/v1",
        api_key="EMPTY",
        temperature=1.0,
        top_p=0.95,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        disable_streaming=True,
        use_responses_api=False,
        extra_body={
            "top_k": 64,
            "include_reasoning": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return _create_subagent(model)
