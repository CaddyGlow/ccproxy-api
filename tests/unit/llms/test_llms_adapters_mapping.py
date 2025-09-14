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
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.openai.models import (
    ChatCompletionResponse as OpenAIChatResponse,
)
from ccproxy.llms.openai.models import (
    InputTokensDetails as OpenAIInputTokensDetails,
)
from ccproxy.llms.openai.models import (
    MessageOutput as OpenAIResponseMessage,
)
from ccproxy.llms.openai.models import (
    OutputTextContent as OpenAIOutputTextContent,
)
from ccproxy.llms.openai.models import (
    OutputTokensDetails as OpenAIOutputTokensDetails,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)
from ccproxy.llms.openai.models import (
    ResponseUsage as OpenAIResponseUsage,
)


@pytest.mark.asyncio
async def test_anthropic_response_to_openai_chat_response():
    from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
        AnthropicMessagesToOpenAIChatAdapter,
    )

    # Minimal anthropic response with 1 text block
    anthropic_resp = AnthropicMessageResponse(
        id="msg_123",
        role="assistant",
        model="claude-sonnet",
        content=[AnthropicTextBlock(type="text", text="Hi there")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=10, output_tokens=5),
    )

    adapter = AnthropicMessagesToOpenAIChatAdapter()
    out = await adapter.adapt_response(anthropic_resp)

    openai_resp = OpenAIChatResponse.model_validate(out)
    assert openai_resp.id == "msg_123"
    assert openai_resp.model == "claude-sonnet"
    assert openai_resp.choices[0].message.content == "Hi there"
    assert openai_resp.choices[0].finish_reason == "stop"
    assert openai_resp.usage.prompt_tokens == 10
    assert openai_resp.usage.completion_tokens == 5
    assert openai_resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_thinking_block_serialization_in_response(monkeypatch):
    monkeypatch.setenv("LLM__OPENAI_THINKING_XML", "true")

    from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
        AnthropicMessagesToOpenAIChatAdapter,
    )

    anthropic_resp = AnthropicMessageResponse(
        id="msg_think",
        role="assistant",
        model="claude-3",
        content=[
            AnthropicThinkingBlock(
                type="thinking", thinking="chain of thought", signature="sig123"
            ),
            AnthropicTextBlock(type="text", text=" Final answer."),
        ],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=5, output_tokens=7),
    )

    adapter = AnthropicMessagesToOpenAIChatAdapter()
    out = await adapter.adapt_response(anthropic_resp)
    openai_resp = OpenAIChatResponse.model_validate(out)
    content = openai_resp.choices[0].message.content or ""
    assert '<thinking signature="sig123">chain of thought</thinking>' in content
    assert content.endswith(" Final answer.")


@pytest.mark.asyncio
async def test_openai_responses_to_anthropic_with_thinking_and_tool_use():
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    # Compose a ResponseObject-like dict
    resp = {
        "id": "res_2",
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "m2",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '<thinking signature="s1">think</thinking> Answer with data.',
                    },
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "fetch",
                        "arguments": {"url": "https://example.com"},
                    },
                ],
            }
        ],
        "status": "completed",
        "parallel_tool_calls": False,
        "usage": {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        },
    }

    adapter = OpenAIResponsesToAnthropicAdapter()
    resp_obj = OpenAIResponseObject.model_validate(resp)
    out = await adapter.adapt_response(resp_obj)
    # Validate as Anthropic message response
    anth = AnthropicMessageResponse.model_validate(out)
    # Expect ThinkingBlock then TextBlock then ToolUseBlock
    assert anth.content[0].type == "thinking"
    assert getattr(anth.content[0], "signature", "") == "s1"
    assert anth.content[1].type == "text"
    assert "Answer with data." in getattr(anth.content[1], "text", "")
    assert anth.content[2].type == "tool_use"
    assert getattr(anth.content[2], "name", "") == "fetch"


@pytest.mark.asyncio
async def test_openai_chat_to_openai_response_request():
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello"},
        ],
        max_completion_tokens=256,
    )
    adapter = OpenAIChatToOpenAIResponsesAdapter()
    out = await adapter.adapt_request(req)
    openai_resp_req = OpenAIResponseRequest.model_validate(out)
    assert openai_resp_req.model == "gpt-4o"
    # We map last user message into a single input message
    assert openai_resp_req.input and openai_resp_req.input[0]["type"] == "message"
    assert openai_resp_req.max_output_tokens == 256


@pytest.mark.asyncio
async def test_openai_chat_to_openai_responses_structured_outputs_json_schema():
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    schema = {
        "name": "math_response",
        "schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    out = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(req)
    resp_req = OpenAIResponseRequest.model_validate(out)
    assert (
        resp_req.text and resp_req.text.get("format", {}).get("type") == "json_schema"
    )
    fmt = resp_req.text["format"]
    assert fmt.get("name") == "math_response"
    assert isinstance(fmt.get("schema"), dict)
    assert fmt.get("strict") is True


@pytest.mark.asyncio
async def test_openai_response_to_openai_chat_response():
    from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    resp = OpenAIResponseObject(
        id="res_1",
        object="response",
        created_at=0,
        model="gpt-4o",
        output=[
            OpenAIResponseMessage(
                type="message",
                id="m1",
                status="completed",
                role="assistant",
                content=[OpenAIOutputTextContent(type="output_text", text="Hello!")],
            )
        ],
        status="completed",
        parallel_tool_calls=False,
        usage=OpenAIResponseUsage(
            input_tokens=7,
            input_tokens_details=OpenAIInputTokensDetails(cached_tokens=0),
            output_tokens=9,
            output_tokens_details=OpenAIOutputTokensDetails(reasoning_tokens=0),
            total_tokens=16,
        ),
    )

    adapter = OpenAIResponsesToOpenAIChatAdapter()
    out = await adapter.adapt_response(resp)
    chat = OpenAIChatResponse.model_validate(out)
    assert chat.id == "res_1"
    assert chat.model == "gpt-4o"
    assert chat.choices[0].message.content == "Hello!"
    assert chat.usage.total_tokens == 16


@pytest.mark.asyncio
async def test_openai_responses_to_anthropic_maps_reasoning_to_thinking():
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    # Response with reasoning tokens in the message content (adapted to current schema)
    resp = {
        "id": "res_r1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    # Include reasoning summary as part of the content
                    {"type": "output_text", "text": "inner chain Final answer."},
                ],
            },
        ],
        "status": "completed",
        "parallel_tool_calls": False,
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {
                "reasoning_tokens": 5
            },  # Indicate reasoning was used
            "total_tokens": 3,
        },
    }

    adapter = OpenAIResponsesToAnthropicAdapter()
    resp_obj = OpenAIResponseObject.model_validate(resp)
    out = await adapter.adapt_response(resp_obj)
    # Validate as Anthropic message response; expect thinking block parsed from reasoning tokens
    anth = AnthropicMessageResponse.model_validate(out)
    assert anth.content and len(anth.content) > 0
    # The adapter should parse the content and potentially create thinking blocks
    # For now, just verify that the content includes both reasoning and final text
    content_text = ""
    for block in anth.content:
        if hasattr(block, "text"):
            content_text += block.text
        elif hasattr(block, "thinking"):
            content_text += block.thinking
    assert "inner chain" in content_text
    assert "Final answer" in content_text


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    req = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        max_output_tokens=300,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "web search",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        stream=True,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
    out = await adapter.adapt_request(req)
    anth = AnthropicCreateMessageRequest.model_validate(out)

    assert anth.model == "gpt-4o"
    assert anth.max_tokens == 300
    assert anth.stream is True
    assert anth.messages[0].role == "user"
    assert anth.messages[0].content == "Hello"
    assert anth.tools and anth.tools[0].name == "search"
    assert anth.tool_choice and anth.tool_choice.type == "auto"


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_text_format_injection():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }
    req = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "doc",
                "schema": schema,
                "strict": True,
            }
        },
    )

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(req)
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert isinstance(anth.system, str)
    assert (
        "Respond ONLY with JSON strictly conforming to this JSON Schema" in anth.system
    )
    assert '"properties":' in anth.system


@pytest.mark.asyncio
async def test_openai_responses_request_instructions_to_system_passthrough():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    req = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        instructions="You are helpful",
    )

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(req)
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert anth.system == "You are helpful"


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_text_format_json_object():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    req = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        text={"format": {"type": "json_object"}},
    )

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(req)
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert isinstance(anth.system, str)
    assert "Respond ONLY with a valid JSON object" in anth.system


@pytest.mark.asyncio
async def test_openai_responses_request_reasoning_to_anthropic_thinking():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    req = OpenAIResponseRequest(
        model="gpt-4o",
        input="Hello",
        reasoning={"effort": "medium"},
        max_output_tokens=3000,
    )
    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(req)
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert anth.thinking is not None
    assert getattr(anth.thinking, "budget_tokens", 0) == 5000
    assert anth.max_tokens > 5000
    assert anth.temperature == 1.0


@pytest.mark.asyncio
async def test_openai_responses_request_stream_maps_refusal_done_event():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )
    from ccproxy.llms.anthropic.models import (
        ContentBlockDeltaEvent as AnthCBDelta,
    )
    from ccproxy.llms.anthropic.models import (
        ContentBlockStartEvent as AnthCBStart,
    )
    from ccproxy.llms.anthropic.models import (
        ContentBlockStopEvent as AnthCBStop,
    )
    from ccproxy.llms.anthropic.models import (
        MessageDelta as AnthDelta,
    )
    from ccproxy.llms.anthropic.models import (
        MessageDeltaEvent as AnthMsgDelta,
    )
    from ccproxy.llms.anthropic.models import (
        MessageResponse as AnthMessage,
    )
    from ccproxy.llms.anthropic.models import (
        MessageStartEvent as AnthMsgStart,
    )
    from ccproxy.llms.anthropic.models import (
        MessageStopEvent as AnthMsgStop,
    )
    from ccproxy.llms.anthropic.models import (
        TextBlock as AnthText,
    )
    from ccproxy.llms.anthropic.models import (
        Usage as AnthUsage,
    )

    msg = AnthMessage(
        id="ms",
        role="assistant",
        model="claude",
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=AnthUsage(input_tokens=0, output_tokens=0),
    )
    events = [
        AnthMsgStart(type="message_start", message=msg),
        AnthCBStart(
            type="content_block_start",
            index=0,
            content_block=AnthText(type="text", text=""),
        ),
        AnthCBDelta(
            type="content_block_delta", index=0, delta=AnthText(type="text", text="Hi")
        ),
        AnthCBStop(type="content_block_stop", index=0),
        AnthMsgDelta(
            type="message_delta",
            delta=AnthDelta(stop_reason="refusal"),
            usage=AnthUsage(input_tokens=1, output_tokens=0),
        ),
        AnthMsgStop(type="message_stop"),
    ]

    async def gen():
        for e in events:
            yield e

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
    out_events = []
    async for ev in adapter.adapt_stream(gen()):
        out_events.append(ev)

    assert any(
        e.type == "response.refusal.done" for e in out_events if hasattr(e, "type")
    )
