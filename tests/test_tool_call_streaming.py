"""Tests for tool call streaming fix in OpenAI stream processor.

This module tests the fix for incomplete tool call responses that were occurring
when using dict output format (SDK mode). The bug was that tool calls with
streaming JSON deltas were only yielded for SSE format, not dict format.
"""

import json
from collections.abc import AsyncIterator

import pytest

from ccproxy.llms.streaming.processors import OpenAIStreamProcessor


async def create_mock_claude_stream_with_tool_call(
    tool_id: str = "tool_123",
    tool_name: str = "get_weather",
    tool_args: dict | None = None,
) -> AsyncIterator[dict]:
    """Create a mock Claude stream with a tool call using streaming JSON deltas.

    This simulates the real Claude SDK streaming format with:
    - message_start
    - content_block_start (tool_use)
    - multiple content_block_delta (input_json_delta) chunks
    - content_block_stop
    - message_delta (with usage)
    - message_stop
    """
    if tool_args is None:
        tool_args = {"location": "San Francisco", "unit": "celsius"}

    # Simulate streaming JSON in chunks (like real Claude API)
    json_str = json.dumps(tool_args)
    chunk_size = 10
    json_chunks = [json_str[i : i + chunk_size] for i in range(0, len(json_str), chunk_size)]

    # Message start
    yield {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    }

    # Content block start - tool use
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
        },
    }

    # Stream JSON arguments in chunks
    for chunk in json_chunks:
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": chunk,
            },
        }

    # Content block stop - this should trigger tool call output
    yield {
        "type": "content_block_stop",
        "index": 0,
    }

    # Message delta with usage
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 50},
    }

    # Message stop
    yield {"type": "message_stop"}


async def create_mock_claude_stream_with_multiple_tools() -> AsyncIterator[dict]:
    """Create a mock Claude stream with multiple tool calls."""
    # Message start
    yield {
        "type": "message_start",
        "message": {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 0},
        },
    }

    # First tool call
    tool1_args = json.dumps({"location": "San Francisco"})
    yield {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tool_1", "name": "get_weather"},
    }
    for chunk in [tool1_args[i : i + 10] for i in range(0, len(tool1_args), 10)]:
        yield {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": chunk},
        }
    yield {"type": "content_block_stop", "index": 0}

    # Second tool call
    tool2_args = json.dumps({"city": "New York", "country": "USA"})
    yield {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "tool_2", "name": "get_population"},
    }
    for chunk in [tool2_args[i : i + 10] for i in range(0, len(tool2_args), 10)]:
        yield {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": chunk},
        }
    yield {"type": "content_block_stop", "index": 1}

    # Message delta and stop
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        "usage": {"output_tokens": 50},
    }
    yield {"type": "message_stop"}


@pytest.mark.asyncio
async def test_tool_call_streaming_dict_format():
    """Test that tool calls are properly streamed in dict format (SDK mode).

    This test verifies the fix for the bug where tool calls with streaming
    JSON deltas were only yielded for SSE format, not dict format.
    """
    # Create processor with dict output format (like SDK mode)
    processor = OpenAIStreamProcessor(
        enable_usage=True,
        enable_tool_calls=True,
        output_format="dict",
    )

    # Process mock stream
    mock_stream = create_mock_claude_stream_with_tool_call()
    chunks = []
    async for chunk in processor.process_stream(mock_stream):
        chunks.append(chunk)

    # Verify we got tool call chunks
    tool_call_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
    ]

    # Should have exactly one tool call chunk
    assert len(tool_call_chunks) == 1, (
        f"Expected 1 tool call chunk, got {len(tool_call_chunks)}. "
        f"This indicates tool calls are not being yielded for dict format!"
    )

    # Verify tool call structure
    tool_call_chunk = tool_call_chunks[0]
    tool_call = tool_call_chunk["choices"][0]["delta"]["tool_calls"][0]

    assert tool_call["id"] == "tool_123"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"

    # Verify arguments are complete JSON
    arguments = tool_call["function"]["arguments"]
    parsed_args = json.loads(arguments)
    assert parsed_args == {"location": "San Francisco", "unit": "celsius"}

    print(f"Tool call properly streamed in dict format")
    print(f"   Tool ID: {tool_call['id']}")
    print(f"   Tool Name: {tool_call['function']['name']}")
    print(f"   Arguments: {arguments}")


@pytest.mark.asyncio
async def test_tool_call_streaming_multiple_tools():
    """Test that multiple tool calls are properly indexed and yielded."""
    processor = OpenAIStreamProcessor(
        enable_usage=True,
        enable_tool_calls=True,
        output_format="dict",
    )

    mock_stream = create_mock_claude_stream_with_multiple_tools()
    chunks = []
    async for chunk in processor.process_stream(mock_stream):
        chunks.append(chunk)

    # Get all tool call chunks
    tool_call_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
    ]

    # Should have exactly two tool call chunks (one per tool)
    assert len(tool_call_chunks) == 2, (
        f"Expected 2 tool call chunks, got {len(tool_call_chunks)}"
    )

    # Verify first tool call
    tool_call_1 = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call_1["index"] == 0
    assert tool_call_1["id"] == "tool_1"
    assert tool_call_1["function"]["name"] == "get_weather"
    args_1 = json.loads(tool_call_1["function"]["arguments"])
    assert args_1 == {"location": "San Francisco"}

    # Verify second tool call
    tool_call_2 = tool_call_chunks[1]["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call_2["index"] == 0
    assert tool_call_2["id"] == "tool_2"
    assert tool_call_2["function"]["name"] == "get_population"
    args_2 = json.loads(tool_call_2["function"]["arguments"])
    assert args_2 == {"city": "New York", "country": "USA"}

    print(f"Multiple tool calls properly yielded separately")
    print(f"   Tool 1: {tool_call_1['function']['name']} (ID: {tool_call_1['id']})")
    print(f"   Tool 2: {tool_call_2['function']['name']} (ID: {tool_call_2['id']})")


@pytest.mark.asyncio
async def test_tool_call_streaming_sse_format_regression():
    """Test that SSE format still works (no regression from fix)."""
    # Create processor with SSE output format
    processor = OpenAIStreamProcessor(
        enable_usage=True,
        enable_tool_calls=True,
        output_format="sse",
    )

    mock_stream = create_mock_claude_stream_with_tool_call()
    chunks = []
    async for chunk in processor.process_stream(mock_stream):
        chunks.append(chunk)

    # For SSE format, chunks should be strings, not dicts
    assert all(isinstance(c, str) for c in chunks), (
        "SSE format should return strings"
    )

    # Parse SSE chunks to find tool call
    tool_call_chunks = []
    for chunk in chunks:
        if chunk.startswith("data: ") and "tool_calls" in chunk:
            # Extract JSON from "data: {...}\n\n" format
            json_str = chunk[6:].strip()
            if json_str and json_str != "[DONE]":
                tool_call_chunks.append(json.loads(json_str))

    # Should have tool call in SSE format
    assert len(tool_call_chunks) > 0, "Expected tool call chunks in SSE format"

    print(f"SSE format still works correctly (no regression)")
    print(f"   Found {len(tool_call_chunks)} tool call chunk(s) in SSE format")


@pytest.mark.asyncio
async def test_tool_call_arguments_complete():
    """Test that tool call arguments are complete, not truncated."""
    # Create a tool with complex arguments
    complex_args = {
        "location": "San Francisco, CA",
        "unit": "celsius",
        "include_forecast": True,
        "days": 7,
        "details": {
            "humidity": True,
            "wind_speed": True,
            "precipitation": True,
        },
        "alerts": ["severe_weather", "air_quality"],
    }

    async def create_stream() -> AsyncIterator[dict]:
        yield {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        }
        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "tool_complex",
                "name": "get_weather_detailed",
            },
        }

        # Stream JSON in very small chunks to test accumulation
        json_str = json.dumps(complex_args)
        for i in range(0, len(json_str), 5):  # 5-char chunks
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json_str[i : i + 5]},
            }

        yield {"type": "content_block_stop", "index": 0}
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 50},
        }
        yield {"type": "message_stop"}

    processor = OpenAIStreamProcessor(
        enable_usage=True,
        enable_tool_calls=True,
        output_format="dict",
    )

    chunks = []
    async for chunk in processor.process_stream(create_stream()):
        chunks.append(chunk)

    tool_call_chunks = [
        c for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
    ]

    assert len(tool_call_chunks) == 1
    tool_call = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    arguments = tool_call["function"]["arguments"]
    parsed_args = json.loads(arguments)

    # Verify all fields are present and correct
    assert parsed_args == complex_args, (
        "Arguments should be complete and match input exactly"
    )

    print(f"Complex tool call arguments are complete")
    print(f"   Argument keys: {list(parsed_args.keys())}")
    print(f"   Total JSON length: {len(arguments)} chars")


if __name__ == "__main__":
    import asyncio

    print("Running tool call streaming fix tests...\n")

    asyncio.run(test_tool_call_streaming_dict_format())
    print()

    asyncio.run(test_tool_call_streaming_multiple_tools())
    print()

    asyncio.run(test_tool_call_streaming_sse_format_regression())
    print()

    asyncio.run(test_tool_call_arguments_complete())
    print()

    print("\nAll tests passed!")
