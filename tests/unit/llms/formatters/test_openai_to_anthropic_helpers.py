import pytest

from ccproxy.llms.formatters.openai_to_anthropic import (
    convert__openai_chat_to_anthropic_message__request,
    convert__openai_responses_to_anthropic_message__request,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


@pytest.mark.asyncio
async def test_openai_chat_request_to_anthropic_messages_basic() -> None:
    req = openai_models.ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "sys"},  # type: ignore[list-item]
            {"role": "user", "content": "Hello"},  # type: ignore[list-item]
        ],
        max_completion_tokens=128,
        temperature=0.2,
        top_p=0.9,
        stream=True,
    )
    out = await convert__openai_chat_to_anthropic_message__request(req)
    anth_req = anthropic_models.CreateMessageRequest.model_validate(out)

    assert anth_req.model
    assert anth_req.max_tokens == 128
    assert anth_req.stream is True
    # System mapped
    assert anth_req.system is not None
    # Last user message content mapped
    assert anth_req.messages and anth_req.messages[0].role == "user"


@pytest.mark.asyncio
async def test_openai_chat_tools_and_choice_mapping() -> None:
    from ccproxy.llms.formatters.openai_to_anthropic import (
        convert__openai_chat_to_anthropic_message__request,
    )

    req = openai_models.ChatCompletionRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "calc"}],  # type: ignore[list-item]
        tools=[
            {  # type: ignore[list-item]
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Calculator",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                    },
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    out = await convert__openai_chat_to_anthropic_message__request(req)
    anth_req = anthropic_models.CreateMessageRequest.model_validate(out)

    assert anth_req.tools and anth_req.tools[0].name == "calc"
    # tool_choice auto should map through to an Anthropic-compatible structure
    assert anth_req.tool_choice is not None


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_basic() -> None:
    resp_req = openai_models.ResponseRequest(
        model="gpt-4o",
        instructions="sys",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        max_output_tokens=64,
    )

    out = convert__openai_responses_to_anthropic_message__request(resp_req)
    anth_req = anthropic_models.CreateMessageRequest.model_validate(out)

    assert anth_req.model
    assert anth_req.max_tokens == 64
    assert anth_req.system == "sys"
    assert anth_req.messages and anth_req.messages[0].role == "user"


@pytest.mark.asyncio
async def test_openai_chat_to_anthropic_supports_max_thinking_effort() -> None:
    req = openai_models.ChatCompletionRequest(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "Think carefully"}],  # type: ignore[list-item]
        max_completion_tokens=128,
        reasoning_effort="max",
    )

    anth_req = await convert__openai_chat_to_anthropic_message__request(req)

    assert anth_req.model == "claude-opus-4-7"
    assert anth_req.thinking is not None
    assert anth_req.thinking.type == "enabled"
    assert anth_req.thinking.budget_tokens == 32000
    assert anth_req.max_tokens == 32064
    assert anth_req.temperature == 1.0


def test_openai_responses_to_anthropic_maps_reasoning_effort() -> None:
    resp_req = openai_models.ResponseRequest(
        model="claude-opus-4-7",
        input="Think carefully",
        max_output_tokens=128,
        reasoning={"effort": "xhigh"},
    )

    anth_req = convert__openai_responses_to_anthropic_message__request(resp_req)

    assert anth_req.model == "claude-opus-4-7"
    assert anth_req.thinking is not None
    assert anth_req.thinking.type == "enabled"
    assert anth_req.thinking.budget_tokens == 20000
    assert anth_req.max_tokens == 20064
    assert anth_req.temperature == 1.0
