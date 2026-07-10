import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.decomposer_agent.app import (
    ChatNeMoGym,
    NeMoGymContext,
    _input_to_messages,
    _messages_to_items,
    _request_with_body,
)


@pytest.mark.parametrize(
    "input_items",
    [
        pytest.param(
            [
                {"type": "message", "role": "system", "content": "system prompt"},
                {"type": "message", "role": "user", "content": "question"},
                {"type": "message", "role": "assistant", "content": "thinking"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": json.dumps({"query": "abc", "limit": 2}),
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "answer",
                    "status": "completed",
                },
            ],
            id="normalized-agent-server-input",
        ),
    ],
)
def test_messages_to_input_round_trips_agent_server_input(input_items):
    assert _messages_to_items(_input_to_messages(input_items)) == input_items


def test_request_with_body_adds_body_to_model_settings():
    body = _body()
    request = _FakeModelRequest(
        context=NeMoGymContext(body=body),
        model_settings={"existing": "setting"},
    )

    updated_request = _request_with_body(request)

    assert updated_request.model_settings["existing"] == "setting"
    assert updated_request.model_settings["nemo_gym_body"] == body


def test_chat_nemo_gym_requires_body():
    model = ChatNeMoGym(server_client=_FakeServerClient(), model_server_name="model")

    with pytest.raises(RuntimeError, match="nemo_gym_body"):
        asyncio.run(model._agenerate([HumanMessage(content="hi")], tools=[_decomposer_tool()]))


def test_chat_nemo_gym_preserves_model_params_and_overrides_tools():
    server_client = _FakeServerClient()
    model = ChatNeMoGym(server_client=server_client, model_server_name="model")
    body = _body()

    asyncio.run(
        model._agenerate(
            [HumanMessage(content="runtime prompt")],
            nemo_gym_body=body,
            tools=[_decomposer_tool()],
        )
    )

    sent_body = server_client.requests[0]["json"]
    sent = sent_body.model_dump(warnings="error")

    assert sent["model"] == "teacher-model"
    assert sent["temperature"] == 0.7
    assert sent["max_output_tokens"] == 123
    assert sent["parallel_tool_calls"] is False
    assert sent["input"] == [{"type": "message", "role": "user", "content": "runtime prompt"}]
    assert sent["tools"] == [_decomposer_tool()]
    assert sent["tool_choice"] == "auto"


class _FakeModelRequest:
    def __init__(self, context, model_settings):
        self.runtime = SimpleNamespace(context=context)
        self.model_settings = model_settings

    def override(self, **kwargs):
        return SimpleNamespace(model_settings=kwargs["model_settings"])


class _FakeServerClient:
    def __init__(self):
        self.requests = []

    async def post(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeModelResponse()


class _FakeModelResponse:
    ok = True
    cookies = {}

    async def read(self):
        return json.dumps(_model_response()).encode()


def _body():
    return NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {
            "input": [{"type": "message", "role": "user", "content": "original prompt"}],
            "model": "teacher-model",
            "temperature": 0.7,
            "max_output_tokens": 123,
            "parallel_tool_calls": False,
            "tools": [_outer_tool()],
            "tool_choice": {"type": "function", "name": "outer_tool"},
        }
    )


def _model_response():
    return {
        "id": "resp_test",
        "created_at": 0.0,
        "model": "teacher-model",
        "object": "response",
        "output": [
            {
                "id": "msg_test",
                "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }


def _outer_tool():
    return {
        "type": "function",
        "name": "outer_tool",
        "description": "Outer resource-server tool that must not leak to Decomposer.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": False,
    }


def _decomposer_tool():
    return {
        "type": "function",
        "name": "spawn_subagent",
        "description": "Spawn a subagent.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "strict": False,
    }
