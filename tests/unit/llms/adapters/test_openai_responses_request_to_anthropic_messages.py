import pytest

from ccproxy.llms.anthropic.models import (
    ContentBlockDeltaEvent as AnthropicContentBlockDeltaEvent,
)
from ccproxy.llms.anthropic.models import (
    ContentBlockStartEvent as AnthropicContentBlockStartEvent,
)
from ccproxy.llms.anthropic.models import (
    ContentBlockStopEvent as AnthropicContentBlockStopEvent,
)
from ccproxy.llms.anthropic.models import (
    CreateMessageRequest as AnthropicCreateMessageRequest,
)
from ccproxy.llms.anthropic.models import (
    MessageDelta as AnthropicMessageDelta,
)
from ccproxy.llms.anthropic.models import (
    MessageDeltaEvent as AnthropicMessageDeltaEvent,
)
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    MessageStartEvent as AnthropicMessageStartEvent,
)
from ccproxy.llms.anthropic.models import (
    MessageStopEvent as AnthropicMessageStopEvent,
)
from ccproxy.llms.anthropic.models import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.anthropic.models import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_adapt_response_delegates() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    anth = AnthropicMessageResponse(
        id="msg_2",
        role="assistant",
        model="claude",
        content=[AnthropicTextBlock(type="text", text="Hello")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=1, output_tokens=2),
    )
    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_response(
        anth.model_dump()
    )
    OpenAIResponseObject.model_validate(out)


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_adapt_stream_minimal() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    msg = AnthropicMessageResponse(
        id="mstream2",
        role="assistant",
        model="claude",
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
    )
    events = [
        AnthropicMessageStartEvent(type="message_start", message=msg).model_dump(),
        AnthropicContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=AnthropicTextBlock(type="text", text=""),
        ).model_dump(),
        AnthropicContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=AnthropicTextBlock(type="text", text="Hi"),
        ).model_dump(),
        AnthropicContentBlockStopEvent(type="content_block_stop", index=0).model_dump(),
        AnthropicMessageDeltaEvent(
            type="message_delta",
            delta=AnthropicMessageDelta(stop_reason="end_turn"),
            usage=AnthropicUsage(input_tokens=1, output_tokens=2),
        ).model_dump(),
        AnthropicMessageStopEvent(type="message_stop").model_dump(),
    ]

    async def gen():
        for e in events:
            yield e

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
    res_events = []
    async for ev in adapter.adapt_stream(gen()):
        res_events.append(ev)

    assert any(
        e.get("type") == "response.output_text.delta" and e.get("delta") == "Hi"
        for e in res_events
    )
    assert any(e.get("type") == "response.completed" for e in res_events)


# Comprehensive request mapping tests


@pytest.mark.asyncio
async def test_openai_responses_request_basic_mapping() -> None:
    """Test basic request mapping from OpenAI Response format to Anthropic Messages"""
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
async def test_complex_input_parameter_handling() -> None:
    """Test various input parameter structures"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test with string input (shorthand)
    req_string = OpenAIResponseRequest(
        model="gpt-4o",
        input="Hello world",
        max_output_tokens=100,
    )
    out_string = await adapter.adapt_request(req_string.model_dump())
    anth_string = AnthropicCreateMessageRequest.model_validate(out_string)
    assert anth_string.messages[0].content == "Hello world"

    # Test with mixed content input
    req_mixed = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "You are helpful"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Look at this"},
                    {
                        "type": "input_image",
                        "image": {"format": "png", "data": "base64data"},
                    },
                ],
            },
        ],
    )
    out_mixed = await adapter.adapt_request(req_mixed.model_dump())
    anth_mixed = AnthropicCreateMessageRequest.model_validate(out_mixed)
    assert anth_mixed.system == "You are helpful"
    assert len(anth_mixed.messages) == 1
    assert isinstance(anth_mixed.messages[0].content, list)
    assert len(anth_mixed.messages[0].content) == 2  # text + image


@pytest.mark.asyncio
async def test_tools_array_processing() -> None:
    """Test comprehensive tools array processing"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test with multiple tools and complex schemas
    complex_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        },
                        "precision": {"type": "integer", "default": 2},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    req_tools = OpenAIResponseRequest(
        model="gpt-4o",
        input="Use tools to help me",
        tools=complex_tools,
        tool_choice={"type": "function", "function": {"name": "calculator"}},
        parallel_tool_calls=False,
    )

    out_tools = await adapter.adapt_request(req_tools.model_dump())
    anth_tools = AnthropicCreateMessageRequest.model_validate(out_tools)

    assert len(anth_tools.tools) == 2
    assert anth_tools.tools[0].name == "calculator"
    assert anth_tools.tools[1].name == "weather"
    assert anth_tools.tool_choice.type == "tool"
    assert getattr(anth_tools.tool_choice, "name", "") == "calculator"
    assert getattr(anth_tools.tool_choice, "disable_parallel_tool_use", False) is True


@pytest.mark.asyncio
async def test_instructions_field_mapping() -> None:
    """Test instructions field mapping to system parameter"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test basic instructions mapping
    req_instructions = OpenAIResponseRequest(
        model="gpt-4o",
        input="Hello",
        instructions="You are a helpful AI assistant. Be concise.",
    )
    out_instructions = await adapter.adapt_request(req_instructions.model_dump())
    anth_instructions = AnthropicCreateMessageRequest.model_validate(out_instructions)
    assert anth_instructions.system == "You are a helpful AI assistant. Be concise."

    # Test instructions with text format injection
    req_both = OpenAIResponseRequest(
        model="gpt-4o",
        input="Generate data",
        instructions="You are helpful",
        text={"format": {"type": "json_object"}},
    )
    out_both = await adapter.adapt_request(req_both.model_dump())
    anth_both = AnthropicCreateMessageRequest.model_validate(out_both)
    # System should contain both instructions and format instruction
    assert "You are helpful" in anth_both.system
    assert "Respond ONLY with a valid JSON object" in anth_both.system


@pytest.mark.asyncio
async def test_text_format_injection_comprehensive() -> None:
    """Test comprehensive text format injection scenarios"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test json_schema format injection
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }
    req_schema = OpenAIResponseRequest(
        model="gpt-4o",
        input="Generate a title",
        text={
            "format": {
                "type": "json_schema",
                "name": "doc",
                "schema": schema,
                "strict": True,
            }
        },
    )
    out_schema = await adapter.adapt_request(req_schema.model_dump())
    anth_schema = AnthropicCreateMessageRequest.model_validate(out_schema)
    assert isinstance(anth_schema.system, str)
    assert (
        "Respond ONLY with JSON strictly conforming to this JSON Schema"
        in anth_schema.system
    )
    assert '"properties":' in anth_schema.system
    assert '"title":' in anth_schema.system

    # Test json_object format injection
    req_json = OpenAIResponseRequest(
        model="gpt-4o",
        input="Generate JSON",
        text={"format": {"type": "json_object"}},
    )
    out_json = await adapter.adapt_request(req_json.model_dump())
    anth_json = AnthropicCreateMessageRequest.model_validate(out_json)
    assert anth_json.system == "Respond ONLY with a valid JSON object."


@pytest.mark.asyncio
async def test_reasoning_parameter_integration() -> None:
    """Test reasoning parameter mapping to thinking"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test reasoning with medium effort
    req_reasoning = OpenAIResponseRequest(
        model="gpt-4o",
        input="Solve this problem",
        reasoning={"effort": "medium"},
        max_output_tokens=3000,
    )
    out_reasoning = await adapter.adapt_request(req_reasoning.model_dump())
    anth_reasoning = AnthropicCreateMessageRequest.model_validate(out_reasoning)

    assert anth_reasoning.thinking is not None
    assert getattr(anth_reasoning.thinking, "budget_tokens", 0) == 5000
    assert anth_reasoning.max_tokens > 5000
    assert anth_reasoning.temperature == 1.0

    # Test reasoning with high effort
    req_high = OpenAIResponseRequest(
        model="gpt-4o",
        input="Complex reasoning task",
        reasoning={"effort": "high"},
        max_output_tokens=1000,
    )
    out_high = await adapter.adapt_request(req_high.model_dump())
    anth_high = AnthropicCreateMessageRequest.model_validate(out_high)
    assert getattr(anth_high.thinking, "budget_tokens", 0) == 10000


@pytest.mark.asyncio
async def test_complex_nested_request_structures() -> None:
    """Test complex nested request structures with multiple features"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test comprehensive request with all features
    comprehensive_req = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "System context"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "User question"},
                    {
                        "type": "input_image",
                        "image": {"format": "jpeg", "data": "image_data"},
                    },
                ],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "input_text", "text": "I need to use tools"},
                    {
                        "type": "input_tool_use",
                        "id": "call_1",
                        "name": "search",
                        "arguments": {"query": "example"},
                    },
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_tool_result",
                        "tool_call_id": "call_1",
                        "result": "result",
                    }
                ],
            },
        ],
        instructions="Follow these instructions carefully",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search function",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ],
        tool_choice="auto",
        reasoning={"effort": "low"},
        text={"format": {"type": "json_object"}},
        max_output_tokens=2000,
        temperature=0.7,
        stream=False,
    )

    out_comprehensive = await adapter.adapt_request(comprehensive_req.model_dump())
    anth_comprehensive = AnthropicCreateMessageRequest.model_validate(out_comprehensive)

    # Verify all aspects are properly mapped
    assert anth_comprehensive.model == "gpt-4o"
    assert "Follow these instructions carefully" in anth_comprehensive.system
    assert "Respond ONLY with a valid JSON object" in anth_comprehensive.system
    assert (
        len(anth_comprehensive.messages) == 3
    )  # user -> assistant -> user (tool result)
    assert anth_comprehensive.tools and len(anth_comprehensive.tools) == 1
    assert anth_comprehensive.tool_choice.type == "auto"
    assert anth_comprehensive.thinking is not None
    assert getattr(anth_comprehensive.thinking, "budget_tokens", 0) == 2500
    assert anth_comprehensive.max_tokens > 2500
    assert anth_comprehensive.temperature == 0.7
    assert anth_comprehensive.stream is False


@pytest.mark.asyncio
async def test_edge_case_input_variations() -> None:
    """Test edge cases in input parameter handling"""
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

    # Test with empty input array
    req_empty = OpenAIResponseRequest(
        model="gpt-4o",
        input=[],
        max_output_tokens=100,
    )
    out_empty = await adapter.adapt_request(req_empty.model_dump())
    anth_empty = AnthropicCreateMessageRequest.model_validate(out_empty)
    assert len(anth_empty.messages) == 0 or anth_empty.messages[0].content == ""

    # Test with only system message input
    req_system_only = OpenAIResponseRequest(
        model="gpt-4o",
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "System only"}],
            }
        ],
        max_output_tokens=100,
    )
    out_system_only = await adapter.adapt_request(req_system_only.model_dump())
    anth_system_only = AnthropicCreateMessageRequest.model_validate(out_system_only)
    assert anth_system_only.system == "System only"
    assert len(anth_system_only.messages) == 0
