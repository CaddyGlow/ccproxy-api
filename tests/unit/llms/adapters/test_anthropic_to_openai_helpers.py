import pytest

from ccproxy.llms.formatters.anthropic_to_openai.helpers import (
    convert__anthropic_message_to_openai_chat__response,
    convert__anthropic_message_to_openai_responses__request,
    convert__anthropic_message_to_openai_responses__stream,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_chat__response_basic() -> None:
    resp = anthropic_models.MessageResponse(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-3",
        content=[anthropic_models.TextBlock(type="text", text="Hello")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=anthropic_models.Usage(input_tokens=1, output_tokens=2),
    )

    out = convert__anthropic_message_to_openai_chat__response(resp)
    assert isinstance(out, openai_models.ChatCompletionResponse)
    assert out.object == "chat.completion"
    assert out.choices and out.choices[0].message.content == "Hello"
    assert out.choices[0].finish_reason == "stop"
    assert out.usage.total_tokens == 3


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__stream_minimal() -> None:
    async def gen():
        yield anthropic_models.MessageStartEvent(
            type="message_start",
            message=anthropic_models.MessageResponse(
                id="m1",
                type="message",
                role="assistant",
                model="claude-3",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=anthropic_models.Usage(input_tokens=0, output_tokens=0),
            ),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            delta=anthropic_models.TextBlock(type="text", text="Hi"),
            index=0,
        )
        yield anthropic_models.MessageDeltaEvent(
            type="message_delta",
            delta=anthropic_models.MessageDelta(stop_reason="end_turn"),
            usage=anthropic_models.Usage(input_tokens=1, output_tokens=2),
        )
        yield anthropic_models.MessageStopEvent(type="message_stop")

    chunks = []
    async for evt in convert__anthropic_message_to_openai_responses__stream(gen()):
        chunks.append(evt)

    # Expect sequence: response.created, text delta, in_progress, completed
    types = [getattr(e, "type", None) for e in chunks]
    assert types[0] == "response.created"
    assert types[1] == "response.output_text.delta"
    assert types[-1] == "response.completed"


@pytest.mark.asyncio
async def test_convert__anthropic_message_to_openai_responses__request_basic() -> None:
    req = anthropic_models.CreateMessageRequest(
        model="claude-3",
        system="sys",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=256,
        stream=True,
    )

    out = convert__anthropic_message_to_openai_responses__request(req)
    resp_req = openai_models.ResponseRequest.model_validate(out)
    assert resp_req.model == "claude-3"
    assert resp_req.max_output_tokens == 256
    assert resp_req.stream is True
    assert resp_req.instructions == "sys"
    assert isinstance(resp_req.input, list) and resp_req.input
