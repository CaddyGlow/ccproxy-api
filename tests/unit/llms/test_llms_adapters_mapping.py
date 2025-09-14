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
async def test_openai_chat_to_anthropic_request_basic():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        max_completion_tokens=100,
    )

    adapter = OpenAIChatToAnthropicMessagesAdapter()
    out = await adapter.adapt_request(req.model_dump())

    # Validates as Anthropic create message request
    anth_req = AnthropicCreateMessageRequest.model_validate(out)

    assert anth_req.model == "gpt-4o"
    assert anth_req.max_tokens == 100
    assert anth_req.system == "you are helpful"
    assert len(anth_req.messages) == 1
    assert anth_req.messages[0].role == "user"
    assert anth_req.messages[0].content == "Hello"


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
    out = await adapter.adapt_response(anthropic_resp.model_dump())

    openai_resp = OpenAIChatResponse.model_validate(out)
    assert openai_resp.id == "msg_123"
    assert openai_resp.model == "claude-sonnet"
    assert openai_resp.choices[0].message.content == "Hi there"
    assert openai_resp.choices[0].finish_reason == "stop"
    assert openai_resp.usage.prompt_tokens == 10
    assert openai_resp.usage.completion_tokens == 5
    assert openai_resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_thinking_block_serialization_in_response():
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
    out = await adapter.adapt_response(anthropic_resp.model_dump())
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
    out = await adapter.adapt_response(resp)
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
    out = await adapter.adapt_request(req.model_dump())
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
    out = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(req.model_dump())
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
    out = await adapter.adapt_response(resp.model_dump())
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

    # Response with a reasoning item (summary) and a message output
    resp = {
        "id": "res_r1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5",
        "output": [
            {
                "type": "reasoning",
                "id": "rs1",
                "summary": [{"type": "summary_text", "text": "inner chain"}],
            },
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": " Final answer."},
                ],
            },
        ],
        "status": "completed",
        "parallel_tool_calls": False,
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        },
    }

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = await adapter.adapt_response(resp)
    # Validate as Anthropic message response; expect first block is thinking with summary text
    anth = AnthropicMessageResponse.model_validate(out)
    assert anth.content and anth.content[0].type == "thinking"
    assert "inner chain" in getattr(anth.content[0], "thinking", "")
    assert anth.content[1].type == "text"


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
    out = await adapter.adapt_request(req.model_dump())
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

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(
        req.model_dump()
    )
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

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(
        req.model_dump()
    )
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

    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(
        req.model_dump()
    )
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
    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(
        req.model_dump()
    )
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
        AnthMsgStart(type="message_start", message=msg).model_dump(),
        AnthCBStart(
            type="content_block_start",
            index=0,
            content_block=AnthText(type="text", text=""),
        ).model_dump(),
        AnthCBDelta(
            type="content_block_delta", index=0, delta=AnthText(type="text", text="Hi")
        ).model_dump(),
        AnthCBStop(type="content_block_stop", index=0).model_dump(),
        AnthMsgDelta(
            type="message_delta",
            delta=AnthDelta(stop_reason="refusal"),
            usage=AnthUsage(input_tokens=1, output_tokens=0),
        ).model_dump(),
        AnthMsgStop(type="message_stop").model_dump(),
    ]

    async def gen():
        for e in events:
            yield e

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
    out_events = []
    async for ev in adapter.adapt_stream(gen()):
        out_events.append(ev)

    assert any(e.get("type") == "response.refusal.done" for e in out_events)


@pytest.mark.asyncio
async def test_thinking_request_defaults_and_effort_mapping():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    # o3 model should default to thinking=enabled with 10000 budget
    req1 = OpenAIChatRequest(
        model="o3-large",
        messages=[{"role": "user", "content": "Hello"}],
    )
    adapter = OpenAIChatToAnthropicMessagesAdapter()
    out1 = await adapter.adapt_request(req1.model_dump())
    anth_req1 = AnthropicCreateMessageRequest.model_validate(out1)
    assert anth_req1.thinking is not None
    assert anth_req1.thinking.type == "enabled"
    assert getattr(anth_req1.thinking, "budget_tokens", 0) == 10000
    assert anth_req1.max_tokens > 10000
    assert anth_req1.temperature == 1.0

    # explicit reasoning_effort=medium should map to 5000 and adjust max_tokens
    req2 = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        reasoning_effort="medium",
        max_completion_tokens=3000,
    )
    out2 = await adapter.adapt_request(req2.model_dump())
    anth_req2 = AnthropicCreateMessageRequest.model_validate(out2)
    assert anth_req2.thinking is not None
    assert getattr(anth_req2.thinking, "budget_tokens", 0) == 5000
    assert anth_req2.max_tokens > 5000
    assert anth_req2.temperature == 1.0


@pytest.mark.asyncio
async def test_openai_chat_to_anthropic_images_data_url():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    adapter = OpenAIChatToAnthropicMessagesAdapter()
    out = await adapter.adapt_request(req.model_dump())
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert isinstance(anth.messages[0].content, list)
    # Expect first text block then an image block with base64 source
    assert anth.messages[0].content[0].type == "text"
    assert anth.messages[0].content[1].type == "image"
    assert anth.messages[0].content[1].source.type == "base64"


@pytest.mark.asyncio
async def test_openai_chat_tools_and_tool_choice_mapping():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
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
        tool_choice={"type": "function", "function": {"name": "search"}},
        parallel_tool_calls=False,
    )

    adapter = OpenAIChatToAnthropicMessagesAdapter()
    out = await adapter.adapt_request(req.model_dump())
    anth = AnthropicCreateMessageRequest.model_validate(out)

    assert anth.tools is not None and len(anth.tools) == 1
    tool = anth.tools[0]
    assert tool.type == "custom"
    assert tool.name == "search"
    assert tool.description == "web search"
    assert isinstance(tool.input_schema, dict)

    assert anth.tool_choice is not None
    assert anth.tool_choice.type == "tool"
    assert getattr(anth.tool_choice, "name", "") == "search"
    # parallel_tool_calls=False -> disable_parallel_tool_use=True
    assert getattr(anth.tool_choice, "disable_parallel_tool_use", False) is True


@pytest.mark.asyncio
async def test_response_format_json_object_injects_system():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    out = await OpenAIChatToAnthropicMessagesAdapter().adapt_request(req.model_dump())
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert isinstance(anth.system, str)
    assert "Respond ONLY with a valid JSON object" in anth.system


@pytest.mark.asyncio
async def test_response_format_json_schema_injects_system_with_schema():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    out = await OpenAIChatToAnthropicMessagesAdapter().adapt_request(req.model_dump())
    anth = AnthropicCreateMessageRequest.model_validate(out)
    assert isinstance(anth.system, str)
    assert "strictly conforms to this JSON Schema" in anth.system
    assert '"properties":' in anth.system
