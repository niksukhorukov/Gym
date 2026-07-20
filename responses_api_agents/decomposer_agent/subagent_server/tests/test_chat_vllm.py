from typing import Any

import graph
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from openai.types.chat import ChatCompletion
from chat_vllm import ChatVLLM


def _model(*, preserve_reasoning: bool) -> ChatVLLM:
    return ChatVLLM(
        model="test",
        api_key="EMPTY",
        base_url="http://127.0.0.1:1/v1",
        max_tokens=2048,
        disable_streaming=True,
        use_responses_api=False,
        preserve_reasoning=preserve_reasoning,
    )


def _tool_call_response(
    *,
    reasoning: str | None = "Use the calculator.",
    finish_reason: str = "tool_calls",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "multiply",
                    "arguments": '{"a": 6, "b": 7}',
                },
            }
        ],
    }
    if reasoning is not None:
        message["reasoning"] = reasoning

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        },
    }


@pytest.mark.parametrize("typed_response", [False, True])
def test_captures_and_replays_tool_call_reasoning(typed_response: bool) -> None:
    model = _model(preserve_reasoning=True)
    response: dict[str, Any] | ChatCompletion = _tool_call_response()
    if typed_response:
        response = ChatCompletion.model_validate(response)

    message = model._create_chat_result(response).generations[0].message

    assert message.additional_kwargs["reasoning"] == "Use the calculator."
    assert message.tool_calls[0]["name"] == "multiply"

    payload = model._get_request_payload(
        [
            HumanMessage("What is 6 * 7?"),
            message,
            ToolMessage(content="42", tool_call_id="call_1"),
        ]
    )

    assert payload["messages"][1]["reasoning"] == "Use the calculator."
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "multiply"
    assert payload["max_completion_tokens"] == 2048


def test_captures_and_replays_final_answer_reasoning() -> None:
    model = _model(preserve_reasoning=True)
    response = _tool_call_response()
    response["choices"][0]["message"] = {
        "role": "assistant",
        "content": "The answer is 42.",
        "reasoning": "Six times seven is 42.",
    }
    response["choices"][0]["finish_reason"] = "stop"

    message = model._create_chat_result(response).generations[0].message

    assert message.content == "The answer is 42."
    assert message.additional_kwargs["reasoning"] == "Six times seven is 42."

    payload = model._get_request_payload(
        [
            HumanMessage("What is 6 * 7?"),
            message,
            HumanMessage("Are you sure?"),
        ]
    )

    assert payload["messages"][1]["reasoning"] == "Six times seven is 42."


def test_does_not_replay_reasoning_when_disabled() -> None:
    preserving_model = _model(preserve_reasoning=True)
    message = preserving_model._create_chat_result(_tool_call_response()).generations[0].message
    model = _model(preserve_reasoning=False)

    payload = model._get_request_payload(
        [
            HumanMessage("What is 6 * 7?"),
            message,
            ToolMessage(content="42", tool_call_id="call_1"),
        ]
    )

    assert "reasoning" not in payload["messages"][1]
    assert "<think>" not in str(payload["messages"][1].get("content"))
    assert payload["messages"][1]["tool_calls"][0]["function"]["name"] == "multiply"


def test_raises_when_completion_limit_is_exhausted() -> None:
    model = _model(preserve_reasoning=False)

    with pytest.raises(RuntimeError, match="exhausted max_completion_tokens"):
        model._create_chat_result(_tool_call_response(finish_reason="length"))


@pytest.mark.parametrize(
    ("factory_name", "preserves_reasoning", "expected_extra_body"),
    [
        (
            "gemma_4_e2b_thinking",
            True,
            {
                "top_k": 64,
                "thinking_token_budget": 8192,
                "repetition_detection": {
                    "max_pattern_size": 20,
                    "min_pattern_size": 3,
                    "min_count": 4,
                },
                "include_reasoning": True,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
        (
            "lfm2_5_1_2b_thinking",
            False,
            {
                "top_k": 50,
                "repetition_penalty": 1.05,
                "thinking_token_budget": 8192,
                "repetition_detection": {
                    "max_pattern_size": 20,
                    "min_pattern_size": 3,
                    "min_count": 4,
                },
                "include_reasoning": True,
            },
        ),
    ],
)
def test_graph_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    preserves_reasoning: bool,
    expected_extra_body: dict[str, Any],
) -> None:
    monkeypatch.setattr(graph, "create_agent", lambda **kwargs: kwargs["model"])

    model = getattr(graph, factory_name)()

    assert isinstance(model, ChatVLLM)
    assert model.max_tokens == 16384
    assert model.disable_streaming is True
    assert model.use_responses_api is False
    assert model.preserve_reasoning is preserves_reasoning
    assert model.extra_body == expected_extra_body
