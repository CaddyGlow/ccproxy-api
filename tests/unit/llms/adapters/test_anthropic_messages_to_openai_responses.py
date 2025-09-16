import pytest

from ccproxy.llms.anthropic.models import (
    CreateMessageRequest as AnthropicCreateMessageRequest,
)
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.anthropic.models import (
    ThinkingBlock as AnthropicThinkingBlock,
)
from ccproxy.llms.anthropic.models import (
    Tool as AnthropicTool,
)
from ccproxy.llms.anthropic.models import (
    ToolChoiceAuto as AnthropicToolChoiceAuto,
)
from ccproxy.llms.anthropic.models import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from ccproxy.llms.anthropic.models import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    OutputTextContent as OpenAIOutputTextContent,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)


@pytest.mark.asyncio
async def test_anthropic_to_openai_responses_request_mapping():
    from ccproxy.llms.adapters.anthropic_to_openai.messages_to_responses import (
        AnthropicMessagesToOpenAIResponsesAdapter,
    )

    # Build Anthropic request with system, user messages, tools, tool_choice
    anth_req = AnthropicCreateMessageRequest(
        model="claude-sonnet",
        system="You are helpful",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "there"},
                ],
            }
        ],
        max_tokens=256,
        stream=True,
        tools=[
            AnthropicTool(
                type="custom",
                name="search",
                description="web",
                input_schema={"type": "object"},
            )
        ],
        tool_choice=AnthropicToolChoiceAuto(
            type="auto", disable_parallel_tool_use=True
        ),
    )

    adapter = AnthropicMessagesToOpenAIResponsesAdapter()
    out = await adapter.adapt_request(anth_req)
    req = OpenAIResponseRequest.model_validate(out)

    assert req.model == "claude-sonnet"
    assert req.stream is True
    assert req.max_output_tokens == 256
    assert req.instructions and "helpful" in req.instructions
    assert req.input and req.input[0]["type"] == "message"
    # Expect concatenated text from the last user message
    parts = req.input[0]["content"]
    assert parts and parts[0]["type"] == "input_text"
    assert "Hello" in parts[0]["text"] and "there" in parts[0]["text"]
    # Tools mapped to function tools
    assert req.tools and req.tools[0]["type"] == "function"
    assert req.tools[0]["function"]["name"] == "search"
    # tool_choice auto with disable_parallel -> parallel_tool_calls False
    assert req.tool_choice == "auto"
    assert req.parallel_tool_calls is False


@pytest.mark.asyncio
async def test_anthropic_to_openai_responses_response_mapping_thinking_and_tool_use():
    from ccproxy.llms.adapters.anthropic_to_openai.messages_to_responses import (
        AnthropicMessagesToOpenAIResponsesAdapter,
    )

    anth_resp = AnthropicMessageResponse(
        id="msg_abc",
        role="assistant",
        model="claude-3",
        content=[
            AnthropicThinkingBlock(type="thinking", thinking="inner", signature="sig1"),
            AnthropicTextBlock(type="text", text=" final."),
            AnthropicToolUseBlock(
                type="tool_use", id="t1", name="fetch", input={"q": 1}
            ),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=2, output_tokens=3),
    )

    adapter = AnthropicMessagesToOpenAIResponsesAdapter()
    out = await adapter.adapt_response(anth_resp)
    resp = OpenAIResponseObject.model_validate(out)

    assert resp.id == "msg_abc"
    assert resp.model == "claude-3"
    assert resp.status == "completed"
    assert resp.output and resp.output[0].type == "message"
    contents = resp.output[0].content
    # First content is an OutputTextContent model containing <thinking> and visible text
    assert isinstance(contents[0], OpenAIOutputTextContent)
    assert contents[0].type == "output_text"
    assert "<thinking" in contents[0].text and "inner" in contents[0].text
    assert "final." in contents[0].text
    # Next content is tool_use mapping
    assert contents[1]["type"] == "tool_use"
    assert contents[1]["name"] == "fetch" and contents[1]["id"] == "t1"
    # Usage mapping
    assert resp.usage and resp.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_anthropic_to_openai_responses_stream_mapping():
    from ccproxy.llms.adapters.anthropic_to_openai.messages_to_responses import (
        AnthropicMessagesToOpenAIResponsesAdapter,
    )
    from ccproxy.llms.anthropic import models as anthropic_models

    async def anthropic_stream():
        yield anthropic_models.MessageStartEvent(
            type="message_start",
            message=anthropic_models.MessageResponse(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-3",
                content=[
                    anthropic_models.ThinkingBlock(
                        type="thinking", thinking="I am thinking.", signature="sig1"
                    )
                ],
                stop_reason=None,
                stop_sequence=None,
                usage=anthropic_models.Usage(input_tokens=10, output_tokens=0),
            ),
        )
        yield anthropic_models.ContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=anthropic_models.TextBlock(type="text", text="Hello"),
        )
        yield anthropic_models.MessageDeltaEvent(
            type="message_delta",
            delta=anthropic_models.MessageDelta(stop_reason="refusal"),
            usage=anthropic_models.Usage(input_tokens=10, output_tokens=5),
        )
        yield anthropic_models.MessageStopEvent(type="message_stop")

    adapter = AnthropicMessagesToOpenAIResponsesAdapter()
    stream = adapter.adapt_stream(anthropic_stream())
    chunks = [chunk async for chunk in stream]

    assert len(chunks) == 6

    assert chunks[0].type == "response.created"
    assert chunks[0].response.model == "claude-3"

    assert chunks[1].type == "response.output_text.delta"
    assert "<thinking" in chunks[1].delta

    assert chunks[2].type == "response.output_text.delta"
    assert chunks[2].delta == "Hello"

    assert chunks[3].type == "response.in_progress"

    assert chunks[4].type == "response.refusal.done"

    assert chunks[5].type == "response.completed"
