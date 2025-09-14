import pytest

from ccproxy.llms.openai import models as openai_models


@pytest.mark.asyncio
async def test_openai_responses_stream_function_call_arguments_to_anthropic_tool_use() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=2,
                item_id="it1",
                output_index=0,
                delta='{"q":',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=3,
                item_id="it1",
                output_index=0,
                delta='"abc"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=4,
                item_id="it1",
                output_index=0,
                arguments='{"q":"abc"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=5,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    assert any(
        e.type == "content_block_start" and e.content_block.type == "tool_use"
        for e in out
    )
    assert any(e.type == "content_block_stop" for e in out)


@pytest.mark.asyncio
async def test_openai_responses_stream_reasoning_summary_to_anthropic_thinking() -> (
    None
):
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-5",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=2,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta="I am thinking",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=3,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta=" about the answer.",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDoneEvent(
                type="response.reasoning_summary_text.done",
                sequence_number=4,
                item_id="it1",
                output_index=0,
                summary_index=0,
                text="I am thinking about the answer.",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=5,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="The answer is 42.",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=6,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-5",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    thinking_block_start = next(
        (
            e
            for e in out
            if e.type == "content_block_start"
            and hasattr(e, "content_block")
            and e.content_block.type == "thinking"
        ),
        None,
    )
    assert thinking_block_start is not None
    assert (
        thinking_block_start.content_block.thinking == "I am thinking about the answer."
    )
    assert any(e.type == "content_block_stop" for e in out)

    text_delta = next(
        (
            e
            for e in out
            if e.type == "content_block_delta"
            and hasattr(e, "delta")
            and e.delta.type == "text"
        ),
        None,
    )
    assert text_delta is not None
    assert text_delta.delta.text == "The answer is 42."


@pytest.mark.asyncio
async def test_multiple_tool_calls_streaming() -> None:
    """Test streaming with multiple parallel tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

        # First tool call
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=2,
                item_id="tool1",
                output_index=0,
                delta='{"query":',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=3,
                item_id="tool1",
                output_index=0,
                delta='"python"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=4,
                item_id="tool1",
                output_index=0,
                arguments='{"query":"python"}',
            )
        )

        # Second tool call
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=5,
                item_id="tool2",
                output_index=1,
                delta='{"expr":',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=6,
                item_id="tool2",
                output_index=1,
                delta='"2+2"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=7,
                item_id="tool2",
                output_index=1,
                arguments='{"expr":"2+2"}',
            )
        )

        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=8,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    # Should have two tool_use content blocks
    tool_starts = [
        e
        for e in out
        if e.type == "content_block_start"
        and hasattr(e, "content_block")
        and e.content_block.type == "tool_use"
    ]
    assert len(tool_starts) == 2

    # Verify tool IDs and inputs - tools use item_id as name
    tool1 = next(
        (t for t in tool_starts if t.content_block.id == "tool1"),
        None,
    )
    tool2 = next(
        (t for t in tool_starts if t.content_block.id == "tool2"),
        None,
    )

    assert tool1 is not None
    assert tool2 is not None
    assert tool1.content_block.input == {"query": "python"}
    assert tool2.content_block.input == {"expr": "2+2"}


@pytest.mark.asyncio
async def test_streaming_error_handling() -> None:
    """Test streaming with various error events"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=2,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="Starting to respond...",
            )
        )

        # Error event
        yield openai_models.AnyStreamEvent(
            root=openai_models.ErrorEvent(
                type="error",
                error=openai_models.ErrorDetail(
                    type="rate_limit_error",
                    message="Rate limit exceeded",
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    # Should handle error gracefully
    assert any(e.type == "message_start" for e in out)
    assert any(e.type == "content_block_delta" for e in out)
    # Error should propagate as an error event
    assert any(e.type == "error" for e in out)


@pytest.mark.asyncio
async def test_tool_call_with_reasoning_streaming() -> None:
    """Test streaming with both reasoning and tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

        # Reasoning first
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=2,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta="I need to search for information",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=3,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta=" about the topic.",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDoneEvent(
                type="response.reasoning_summary_text.done",
                sequence_number=4,
                item_id="it1",
                output_index=0,
                summary_index=0,
                text="I need to search for information about the topic.",
            )
        )

        # Then tool call
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=5,
                item_id="tool1",
                output_index=0,
                delta='{"query":',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=6,
                item_id="tool1",
                output_index=0,
                delta='"machine learning"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=7,
                item_id="tool1",
                output_index=0,
                arguments='{"query":"machine learning"}',
            )
        )

        # Then text response
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=8,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="Based on the search, ",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=9,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="machine learning is...",
            )
        )

        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=10,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    # Should have thinking block, then tool use block, then text block
    content_starts = [
        e
        for e in out
        if e.type == "content_block_start" and hasattr(e, "content_block")
    ]

    assert len(content_starts) == 3

    thinking_block = content_starts[0].content_block
    tool_block = content_starts[1].content_block
    text_block = content_starts[2].content_block

    assert thinking_block.type == "thinking"
    assert (
        "I need to search for information about the topic." in thinking_block.thinking
    )

    assert tool_block.type == "tool_use"
    assert tool_block.name == "tool1"  # Uses item_id as name
    assert tool_block.input["query"] == "machine learning"

    assert text_block.type == "text"


@pytest.mark.asyncio
async def test_streaming_with_incomplete_tool_calls() -> None:
    """Test handling of incomplete/interrupted tool calls"""
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

        # Start tool call but don't finish it
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=2,
                item_id="incomplete_tool",
                output_index=0,
                delta='{"query":',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=3,
                item_id="incomplete_tool",
                output_index=0,
                delta='"partial',
            )
        )
        # Stream ends abruptly without function_call_arguments.done
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseIncompleteEvent(
                type="response.incomplete",
                sequence_number=4,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="incomplete",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    # Should handle incomplete tool call gracefully
    assert any(e.type == "message_start" for e in out)
    # May have partial tool use block
    tool_starts = [
        e
        for e in out
        if e.type == "content_block_start"
        and hasattr(e, "content_block")
        and e.content_block.type == "tool_use"
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
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_123",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )

        # Initial text
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=2,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="Let me help you with that. ",
            )
        )

        # Reasoning
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=3,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta="First I should search",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=4,
                item_id="it1",
                output_index=0,
                summary_index=0,
                delta=" for relevant information.",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ReasoningSummaryTextDoneEvent(
                type="response.reasoning_summary_text.done",
                sequence_number=5,
                item_id="it1",
                output_index=0,
                summary_index=0,
                text="First I should search for relevant information.",
            )
        )

        # Tool call
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDeltaEvent(
                type="response.function_call_arguments.delta",
                sequence_number=6,
                item_id="search_tool",
                output_index=1,
                delta='{"q": "python basics"}',
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseFunctionCallArgumentsDoneEvent(
                type="response.function_call_arguments.done",
                sequence_number=7,
                item_id="search_tool",
                output_index=1,
                arguments='{"q": "python basics"}',
            )
        )

        # More text
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=8,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="Based on my search, ",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=9,
                item_id="it1",
                output_index=0,
                content_index=0,
                delta="Python is a programming language.",
            )
        )

        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=10,
                response=openai_models.ResponseObject(
                    id="resp_123",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                    usage=openai_models.ResponseUsage(
                        input_tokens=10,
                        output_tokens=25,
                        total_tokens=35,
                        input_tokens_details=openai_models.InputTokensDetails(
                            cached_tokens=0
                        ),
                        output_tokens_details=openai_models.OutputTokensDetails(
                            reasoning_tokens=0
                        ),
                    ),
                ),
            )
        )

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream_typed(gen()):
        out.append(ev)

    # Verify message structure
    assert any(e.type == "message_start" for e in out)
    assert any(e.type == "message_stop" for e in out)

    # Should have multiple content blocks: text, thinking, tool_use, text
    content_starts = [
        e
        for e in out
        if e.type == "content_block_start" and hasattr(e, "content_block")
    ]
    assert len(content_starts) >= 3  # At least text, thinking, tool_use

    # Verify content block types
    block_types = [e.content_block.type for e in content_starts]
    assert "text" in block_types
    assert "tool_use" in block_types

    # Verify usage information is preserved
    message_stop = next((e for e in out if e.type == "message_stop"), None)
    assert message_stop is not None
