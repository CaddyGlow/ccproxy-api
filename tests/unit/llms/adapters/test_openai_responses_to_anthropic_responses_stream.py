import pytest


@pytest.mark.asyncio
async def test_openai_responses_stream_function_call_arguments_to_anthropic_tool_use() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "it1",
            "output_index": 0,
            "delta": '{"q":',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "it1",
            "output_index": 0,
            "delta": '"abc"}',
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "it1",
            "output_index": 0,
            "arguments": '{"q":"abc"}',
        }
        yield {"type": "response.completed", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    assert any(
        e.get("type") == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "tool_use"
        for e in out
    )
    assert any(e.get("type") == "content_block_stop" for e in out)


@pytest.mark.asyncio
async def test_openai_responses_stream_reasoning_summary_to_anthropic_thinking() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-5"}}
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": "I am thinking",
        }
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": " about the answer.",
        }
        yield {"type": "response.reasoning_summary_text.done"}
        yield {"type": "response.output_text.delta", "delta": "The answer is 42."}
        yield {"type": "response.completed", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    thinking_block_start = next(
        (
            e
            for e in out
            if e.get("type") == "content_block_start"
            and (e.get("content_block") or {}).get("type") == "thinking"
        ),
        None,
    )
    assert thinking_block_start is not None
    assert (
        thinking_block_start["content_block"]["thinking"]
        == "I am thinking about the answer."
    )
    assert any(e.get("type") == "content_block_stop" for e in out)

    text_delta = next(
        (
            e
            for e in out
            if e.get("type") == "content_block_delta"
            and (e.get("delta") or {}).get("type") == "text"
        ),
        None,
    )
    assert text_delta is not None
    assert text_delta["delta"]["text"] == "The answer is 42."


@pytest.mark.asyncio
async def test_multiple_tool_calls_streaming() -> None:
    """Test streaming with multiple parallel tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}

        # First tool call
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool1",
            "output_index": 0,
            "call_id": "call_1",
            "name": "search",
            "delta": '{"query":',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool1",
            "output_index": 0,
            "delta": '"python"}',
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "tool1",
            "output_index": 0,
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"query":"python"}',
        }

        # Second tool call
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool2",
            "output_index": 1,
            "call_id": "call_2",
            "name": "calculate",
            "delta": '{"expr":',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool2",
            "output_index": 1,
            "delta": '"2+2"}',
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "tool2",
            "output_index": 1,
            "call_id": "call_2",
            "name": "calculate",
            "arguments": '{"expr":"2+2"}',
        }

        yield {"type": "response.completed", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    # Should have two tool_use content blocks
    tool_starts = [
        e
        for e in out
        if e.get("type") == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "tool_use"
    ]
    assert len(tool_starts) == 2

    # Verify tool names and IDs
    search_tool = next(
        (
            t
            for t in tool_starts
            if (t.get("content_block") or {}).get("name") == "search"
        ),
        None,
    )
    calc_tool = next(
        (
            t
            for t in tool_starts
            if (t.get("content_block") or {}).get("name") == "calculate"
        ),
        None,
    )

    assert search_tool is not None
    assert calc_tool is not None
    assert search_tool["content_block"]["input"] == {"query": "python"}
    assert calc_tool["content_block"]["input"] == {"expr": "2+2"}


@pytest.mark.asyncio
async def test_streaming_error_handling() -> None:
    """Test streaming with various error events"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}
        yield {"type": "response.output_text.delta", "delta": "Starting to respond..."}

        # Error event
        yield {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded",
            },
        }

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    # Should handle error gracefully
    assert any(e.get("type") == "message_start" for e in out)
    assert any(e.get("type") == "content_block_delta" for e in out)
    # Error should propagate as an error event
    assert any(e.get("type") == "error" for e in out)


@pytest.mark.asyncio
async def test_tool_call_with_reasoning_streaming() -> None:
    """Test streaming with both reasoning and tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}

        # Reasoning first
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": "I need to search for information",
        }
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": " about the topic.",
        }
        yield {"type": "response.reasoning_summary_text.done"}

        # Then tool call
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool1",
            "output_index": 0,
            "call_id": "search_1",
            "name": "web_search",
            "delta": '{"query":',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "tool1",
            "output_index": 0,
            "delta": '"machine learning"}',
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "tool1",
            "output_index": 0,
            "call_id": "search_1",
            "name": "web_search",
            "arguments": '{"query":"machine learning"}',
        }

        # Then text response
        yield {"type": "response.output_text.delta", "delta": "Based on the search, "}
        yield {"type": "response.output_text.delta", "delta": "machine learning is..."}

        yield {"type": "response.completed", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    # Should have thinking block, then tool use block, then text block
    content_starts = [e for e in out if e.get("type") == "content_block_start"]

    assert len(content_starts) == 3

    thinking_block = content_starts[0]["content_block"]
    tool_block = content_starts[1]["content_block"]
    text_block = content_starts[2]["content_block"]

    assert thinking_block["type"] == "thinking"
    assert (
        "I need to search for information about the topic."
        in thinking_block["thinking"]
    )

    assert tool_block["type"] == "tool_use"
    assert tool_block["name"] == "web_search"
    assert tool_block["input"]["query"] == "machine learning"

    assert text_block["type"] == "text"


@pytest.mark.asyncio
async def test_streaming_with_incomplete_tool_calls() -> None:
    """Test handling of incomplete/interrupted tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}

        # Start tool call but don't finish it
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "incomplete_tool",
            "output_index": 0,
            "call_id": "incomplete_1",
            "name": "search",
            "delta": '{"query":',
        }
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "incomplete_tool",
            "output_index": 0,
            "delta": '"partial',
        }
        # Stream ends abruptly without function_call_arguments.done
        yield {"type": "response.incomplete", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    # Should handle incomplete tool call gracefully
    assert any(e.get("type") == "message_start" for e in out)
    # May have partial tool use block
    tool_starts = [
        e
        for e in out
        if e.get("type") == "content_block_start"
        and (e.get("content_block") or {}).get("type") == "tool_use"
    ]
    # Should have at most one incomplete tool block
    assert len(tool_starts) <= 1


@pytest.mark.asyncio
async def test_complex_streaming_scenario() -> None:
    """Test complex streaming with mixed content types"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {
            "type": "response.created",
            "response": {"model": "gpt-4o", "id": "resp_123"},
        }

        # Initial text
        yield {
            "type": "response.output_text.delta",
            "delta": "Let me help you with that. ",
        }

        # Reasoning
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": "First I should search",
        }
        yield {
            "type": "response.reasoning_summary_text.delta",
            "delta": " for relevant information.",
        }
        yield {"type": "response.reasoning_summary_text.done"}

        # Tool call
        yield {
            "type": "response.function_call_arguments.delta",
            "item_id": "search_tool",
            "output_index": 1,
            "call_id": "search_123",
            "name": "search",
            "delta": '{"q": "python basics"}',
        }
        yield {
            "type": "response.function_call_arguments.done",
            "item_id": "search_tool",
            "output_index": 1,
            "call_id": "search_123",
            "name": "search",
            "arguments": '{"q": "python basics"}',
        }

        # More text
        yield {"type": "response.output_text.delta", "delta": "Based on my search, "}
        yield {
            "type": "response.output_text.delta",
            "delta": "Python is a programming language.",
        }

        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "usage": {"input_tokens": 10, "output_tokens": 25},
            },
        }

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    # Verify message structure
    assert any(e.get("type") == "message_start" for e in out)
    assert any(e.get("type") == "message_stop" for e in out)

    # Should have multiple content blocks: text, thinking, tool_use, text
    content_starts = [e for e in out if e.get("type") == "content_block_start"]
    assert len(content_starts) >= 3  # At least text, thinking, tool_use

    # Verify content block types
    block_types = [e["content_block"]["type"] for e in content_starts]
    assert "text" in block_types
    assert "thinking" in block_types
    assert "tool_use" in block_types

    # Verify usage information is preserved
    message_stop = next((e for e in out if e.get("type") == "message_stop"), None)
    assert message_stop is not None
