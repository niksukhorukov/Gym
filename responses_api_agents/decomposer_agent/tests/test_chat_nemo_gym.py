import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import Response
from langchain_core.messages import AIMessage, HumanMessage

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.decomposer_agent.app import (
    ChatNeMoGym,
    DecomposerAgent,
    DecomposerAgentConfig,
    DecomposerAgentResponse,
    DecomposerAgentRunRequest,
    NeMoGymContext,
    _collect_subagent_tool_calls,
    _default_response_for_verifier_factory,
    _input_to_messages,
    _messages_to_items,
    _request_with_body,
    _subagent_tool_calls_and_final_message,
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


def test_input_to_messages_joins_system_and_user_prompts_when_enabled():
    input_items = [
        {"type": "message", "role": "system", "content": "system prompt"},
        {"type": "message", "role": "user", "content": "user prompt"},
    ]

    assert _messages_to_items(_input_to_messages(input_items)) == input_items
    messages = _input_to_messages(
        input_items,
        join_gym_system_and_user_prompts=True,
    )
    assert messages == [HumanMessage(content="system prompt\n\nuser prompt")]


def test_request_with_body_adds_body_to_model_settings():
    body = _body()
    request = _FakeModelRequest(
        context=NeMoGymContext(
            body=body.model_dump(mode="json"),
            resource_server_url="http://resources.test:8080",
            resource_server_cookies={"session": "cookie"},
        ),
        model_settings={"existing": "setting"},
    )

    updated_request = _request_with_body(request)

    assert updated_request.model_settings["existing"] == "setting"
    assert updated_request.model_settings["nemo_gym_body"] == body


def test_collect_subagent_tool_calls_preserves_all_calls_in_report_order():
    final_state = {
        "subagent_runs": {
            "run_a": {
                "subagent_run_id": "run_a",
                "report_sequence_number": 1,
                "tool_calls": [
                    {"id": "call_1", "name": "outer_tool", "args": {"value": "a"}},
                ]
            },
            "run_b": {
                "subagent_run_id": "run_b",
                "report_sequence_number": 0,
                "tool_calls": [
                    {"id": "call_1", "name": "outer_tool", "args": {"value": "b1"}},
                    {"id": "call_2", "name": "not_allowed", "args": {"value": "b2"}},
                ]
            },
        },
    }

    result = _collect_subagent_tool_calls(final_state)

    assert [item.name for item in result] == [
        "outer_tool",
        "not_allowed",
        "outer_tool",
    ]
    assert [item.arguments for item in result] == [
        json.dumps({"value": "b1"}),
        json.dumps({"value": "b2"}),
        json.dumps({"value": "a"}),
    ]
    assert [item.call_id for item in result] == [
        "run_b_call_1",
        "run_b_call_2",
        "run_a_call_1",
    ]


def test_default_response_for_verifier_factory_returns_nemo_gym_response():
    nemo_gym_response = NeMoGymResponse.model_validate(_model_response())

    assert (
        _default_response_for_verifier_factory(
            nemo_gym_response,
            {"messages": [], "subagent_runs": {}},
            _body(),
        )
        is nemo_gym_response
    )


def test_subagent_tool_calls_and_final_message():
    response_data = _model_response()
    response_data["output"] = [
        {
            "type": "function_call",
            "name": "spawn_subagent",
            "call_id": "spawn_1",
            "arguments": "{}",
            "status": "completed",
        },
        response_data["output"][-1],
    ]
    response_data["tools"] = [_decomposer_tool()]
    canonical_response = NeMoGymResponse.model_validate(response_data)

    verifier_response = _subagent_tool_calls_and_final_message(
        canonical_response,
        _final_state(),
        _body(),
    )

    assert [item.type for item in canonical_response.output] == ["function_call", "message"]
    assert [item.type for item in verifier_response.output] == ["function_call", "message"]
    assert verifier_response.output[0].name == "outer_tool"
    assert verifier_response.output[-1].content[0].text == "done"
    assert [tool.name for tool in verifier_response.tools] == ["outer_tool"]
    assert verifier_response.tool_choice.name == "outer_tool"
    assert verifier_response.parallel_tool_calls is False


def test_decomposer_agent_response_serializes_final_state():
    response = DecomposerAgentResponse.model_validate(
        _model_response()
        | {
            "final_state": {
                "messages": [HumanMessage(content="question")],
                "subagent_runs": {},
            }
        }
    )

    final_state = response.model_dump(mode="json")["final_state"]

    assert final_state["messages"][0]["type"] == "human"
    assert final_state["messages"][0]["content"] == "question"
    assert final_state["subagent_runs"] == {}


def test_factories_can_be_imported_from_config():
    config_data = {
        "host": "127.0.0.1",
        "port": 8000,
        "entrypoint": "app.py",
        "name": "decomposer",
        "resources_server": {"type": "resources_servers", "name": "resources"},
        "model_server": {"type": "responses_api_models", "name": "model"},
        "subagent_types": [],
    }
    default_config = DecomposerAgentConfig.model_validate(config_data)
    config = DecomposerAgentConfig.model_validate(
        config_data
        | {
            "few_shot_message_factories": ["builtins:list"],
            "response_for_verifier_factory": (
                "responses_api_agents.decomposer_agent.app:"
                "_subagent_tool_calls_and_final_message"
            ),
        }
    )

    assert default_config.few_shot_message_factories == ()
    assert default_config.response_for_verifier_factory is _default_response_for_verifier_factory
    assert config.few_shot_message_factories == [list]
    assert (
        config.response_for_verifier_factory
        is _subagent_tool_calls_and_final_message
    )


def test_run_verifies_subagent_calls_and_returns_canonical_response_with_final_state():
    factory_calls = []

    def response_for_verifier_factory(*args):
        factory_calls.append(args)
        return _subagent_tool_calls_and_final_message(*args)

    response_data = _model_response()
    response_data["output"] = [
        {
            "type": "function_call",
            "name": "spawn_subagent",
            "call_id": "spawn_1",
            "arguments": "{}",
            "status": "completed",
        },
        response_data["output"][-1],
    ]
    response_data["tools"] = [_decomposer_tool()]
    response_data["final_state"] = _final_state()
    server_client = _FakeRunServerClient(response_data)
    agent = DecomposerAgent.model_construct(
        config=SimpleNamespace(
            name="decomposer",
            resources_server=SimpleNamespace(name="resources"),
            response_for_verifier_factory=response_for_verifier_factory,
        ),
        server_client=server_client,
    )

    result = asyncio.run(
        agent.run(
            SimpleNamespace(cookies={"initial": "cookie"}),
            DecomposerAgentRunRequest(responses_create_params=_body()),
        )
    )

    verifier_response = server_client.verify_request["response"]
    assert len(factory_calls) == 1
    assert factory_calls[0][1] == _final_state()
    assert [item["type"] for item in verifier_response["output"]] == ["function_call", "message"]
    assert verifier_response["output"][0]["name"] == "outer_tool"
    assert [tool["name"] for tool in verifier_response["tools"]] == ["outer_tool"]

    assert [item.type for item in result.response.output] == ["function_call", "message"]
    assert result.response.output[0].name == "spawn_subagent"
    assert [tool.name for tool in result.response.tools] == ["spawn_subagent"]
    result_data = result.model_dump(mode="json")
    assert "final_state" not in result_data["response"]
    assert result_data["decomposer_final_state"] == _final_state()


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


def test_responses_omits_unset_body_fields_from_runtime_context():
    graph = _FakeGraph()
    agent = _ResponsesTestAgent.model_construct(
        config=SimpleNamespace(
            resources_server=SimpleNamespace(name="resources"),
            few_shot_message_factories=(),
            join_gym_system_and_user_prompts=False,
        ),
        graph=graph,
    )
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": "runtime prompt"}
    )

    asyncio.run(
        agent.responses(
            SimpleNamespace(cookies={}),
            Response(),
            body,
        )
    )

    assert graph.context["body"] == {"input": "runtime prompt"}
    assert graph.state == {"messages": [HumanMessage(content="runtime prompt")]}


def test_responses_prepends_configured_few_shot_messages():
    graph = _FakeGraph()
    few_shot_messages = [{"role": "user", "content": "few-shot prompt"}]
    agent = _ResponsesTestAgent.model_construct(
        config=SimpleNamespace(
            resources_server=SimpleNamespace(name="resources"),
            few_shot_message_factories=(lambda: few_shot_messages,),
            join_gym_system_and_user_prompts=False,
        ),
        graph=graph,
    )
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        {"input": "runtime prompt"}
    )

    asyncio.run(
        agent.responses(
            SimpleNamespace(cookies={}),
            Response(),
            body,
        )
    )

    assert graph.state == {
        "messages": [
            *few_shot_messages,
            HumanMessage(content="runtime prompt"),
        ]
    }


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


class _FakeRunServerClient:
    def __init__(self, decomposer_response):
        self.decomposer_response = decomposer_response
        self.verify_request = None

    async def post(self, *, server_name, url_path, json, cookies):
        if url_path == "/seed_session":
            return _FakeJSONResponse({}, {"session": "seeded"})
        if url_path == "/v1/responses":
            return _FakeJSONResponse(self.decomposer_response, {"session": "agent"})
        if url_path == "/verify":
            self.verify_request = json
            return _FakeJSONResponse(json | {"reward": 1.0}, cookies)
        raise AssertionError(f"Unexpected request: {server_name=} {url_path=}")


class _FakeGraph:
    context = None
    state = None

    async def ainvoke(self, state, *, context):
        self.state = state
        self.context = context
        return {
            "messages": [
                *state["messages"],
                AIMessage(
                    content="done",
                    response_metadata={
                        "nemo_gym_response": _model_response(),
                        "nemo_gym_cookies": {},
                    },
                ),
            ]
        }


class _ResponsesTestAgent(DecomposerAgent):
    def _resources_server_base_url(self):
        return "http://resources.test"


class _FakeJSONResponse:
    ok = True

    def __init__(self, data, cookies):
        self.data = data
        self.cookies = cookies

    async def read(self):
        return json.dumps(self.data).encode()


def _final_state():
    return {
        "messages": [],
        "subagent_runs": {
            "run_a": {
                "subagent_run_id": "run_a",
                "report_sequence_number": 0,
                "tool_calls": [
                    {"id": "call_1", "name": "outer_tool", "args": {"value": "a"}},
                ],
            }
        },
    }


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
