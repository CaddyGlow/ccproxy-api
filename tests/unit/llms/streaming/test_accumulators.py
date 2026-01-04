"""Unit tests for stream accumulators."""

from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import (
    ClaudeAccumulator,
    OpenAIAccumulator,
    ResponsesAccumulator,
)


def test_claude_accumulator_rebuild_response() -> None:
    """Test that ClaudeAccumulator can rebuild a response object."""
    # Create a Claude accumulator
    accumulator = ClaudeAccumulator()

    # Mock some content block events
    accumulator.accumulate(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "id": "block_1",
                "type": "text",
            },
        },
    )

    # Add text content via delta
    accumulator.accumulate(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello",
            },
        },
    )

    # Add more text content
    accumulator.accumulate(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": " world",
            },
        },
    )

    # End the content block
    accumulator.accumulate(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": 0,
        },
    )

    # Mock a tool use block
    accumulator.accumulate(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "id": "block_2",
                "type": "tool_use",
                "name": "test_tool",
                "input": {},
            },
        },
    )

    # Add tool input via JSON delta
    accumulator.accumulate(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"foo": "bar"}',
            },
        },
    )

    # End the tool block
    accumulator.accumulate(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": 1,
        },
    )

    # Create a mock Claude response
    original_response = {
        "id": "msg_123",
        "model": "claude-3-5-sonnet-20240620",
        "type": "message",
        "role": "assistant",
        "content": [],  # Empty content to be filled
    }

    # Rebuild the response
    rebuilt = accumulator.rebuild_response_object(original_response)

    # Verify the rebuilt response
    assert rebuilt["id"] == original_response["id"]
    assert rebuilt["model"] == original_response["model"]
    assert rebuilt["type"] == original_response["type"]
    assert rebuilt["role"] == original_response["role"]
    assert len(rebuilt["content"]) == 2

    # Verify text block
    text_block = rebuilt["content"][0]
    assert text_block["type"] == "text"
    assert text_block["text"] == "Hello world"

    # Verify tool use block
    tool_block = rebuilt["content"][1]
    assert tool_block["type"] == "tool_use"
    assert tool_block["name"] == "test_tool"
    assert tool_block["input"] == {"foo": "bar"}

    # Verify text extraction
    assert "text" in rebuilt
    assert rebuilt["text"] == "Hello world"


def test_openai_accumulator_rebuild_response() -> None:
    """Test that OpenAIAccumulator can rebuild a response object."""
    # Create an OpenAI accumulator
    accumulator = OpenAIAccumulator()

    # Mock some choice events with deltas
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                }
            ]
        },
    )

    # Add content
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                }
            ]
        },
    )

    # Add more content
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " world"},
                }
            ]
        },
    )

    # Add tool call start
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tool_call_1",
                                "type": "function",
                                "function": {"name": "test_function"},
                            }
                        ]
                    },
                }
            ]
        },
    )

    # Add tool call arguments
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"foo":'},
                            }
                        ]
                    },
                }
            ]
        },
    )

    # Add more tool call arguments
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"bar"}'},
                            }
                        ]
                    },
                }
            ]
        },
    )

    # Add finish reason
    accumulator.accumulate(
        "",
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                }
            ]
        },
    )

    # Create a mock OpenAI response
    original_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1694268190,
        "model": "gpt-3.5-turbo-0613",
        "choices": [],  # Empty choices to be filled
    }

    # Rebuild the response
    rebuilt = accumulator.rebuild_response_object(original_response)

    # Verify the rebuilt response
    assert rebuilt["id"] == original_response["id"]
    assert rebuilt["object"] == original_response["object"]
    assert rebuilt["created"] == original_response["created"]
    assert rebuilt["model"] == original_response["model"]
    assert len(rebuilt["choices"]) == 1

    # Verify choice
    choice = rebuilt["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello world"

    # Verify tool calls
    assert "tool_calls" in choice["message"]
    assert len(choice["message"]["tool_calls"]) == 1
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["id"] == "tool_call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "test_function"
    assert tool_call["function"]["arguments"] == '{"foo":"bar"}'


def test_responses_accumulator_rebuild_response() -> None:
    """Test that ResponsesAccumulator can rebuild a response object."""
    # Create a Responses accumulator
    accumulator = ResponsesAccumulator()

    # Mock text item event
    text_added = openai_models.ResponseOutputItemAddedEvent(
        sequence_number=1,
        type="response.output_item.added",
        output_index=0,
        item=openai_models.OutputItem(
            id="item_1",
            type="text",
            text="Hello",
            status="in_progress",
        ),
    )
    accumulator.accumulate(text_added.type, text_added.model_dump())

    # Mock text item completion
    text_done = openai_models.ResponseOutputItemDoneEvent(
        sequence_number=2,
        type="response.output_item.done",
        output_index=0,
        item=openai_models.OutputItem(
            id="item_1",
            type="text",
            text="Hello world",
            status="completed",
        ),
    )
    accumulator.accumulate(text_done.type, text_done.model_dump())

    # Mock function call item
    call_added = openai_models.ResponseOutputItemAddedEvent(
        sequence_number=3,
        type="response.output_item.added",
        output_index=1,
        item=openai_models.OutputItem(
            id="item_2",
            type="function_call",
            name="test_function",
            status="in_progress",
            call_id="call_1",
        ),
    )
    accumulator.accumulate(call_added.type, call_added.model_dump())

    # Mock function arguments delta
    args_delta_first = openai_models.ResponseFunctionCallArgumentsDeltaEvent(
        sequence_number=4,
        type="response.function_call_arguments.delta",
        item_id="item_2",
        output_index=1,
        delta='{"foo":"',
    )
    accumulator.accumulate(args_delta_first.type, args_delta_first.model_dump())

    # Mock more function arguments delta
    args_delta_second = openai_models.ResponseFunctionCallArgumentsDeltaEvent(
        sequence_number=5,
        type="response.function_call_arguments.delta",
        item_id="item_2",
        output_index=1,
        delta='bar"}',
    )
    accumulator.accumulate(args_delta_second.type, args_delta_second.model_dump())

    # Mock function call completion
    call_done = openai_models.ResponseOutputItemDoneEvent(
        sequence_number=6,
        type="response.output_item.done",
        output_index=1,
        item=openai_models.OutputItem(
            id="item_2",
            type="function_call",
            name="test_function",
            arguments='{"foo":"bar"}',
            status="completed",
            call_id="call_1",
        ),
    )
    accumulator.accumulate(call_done.type, call_done.model_dump())

    # Create a mock Responses API response
    original_response = {
        "id": "resp_123",
        "model": "gpt-4o-2024-05-13",
        "created_at": 1687445455,
        "status": "in_progress",
        "output": {
            "type": "responses",
            "id": "output_123",
        },
    }

    # Rebuild the response
    rebuilt = accumulator.rebuild_response_object(original_response)

    # Verify the rebuilt response
    assert rebuilt["id"] == original_response["id"]
    assert rebuilt["model"] == original_response["model"]
    assert rebuilt["created_at"] == original_response["created_at"]
    assert rebuilt["status"] == original_response["status"]

    # Verify output
    assert "output" in rebuilt
    output_items = rebuilt["output"]
    assert isinstance(output_items, list)

    # Verify function calls metadata
    assert "tool_calls" in rebuilt
    function_call = rebuilt["tool_calls"][0]
    assert function_call["id"] == "item_2"
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == "call_1"
    assert function_call["function"]["name"] == "test_function"
    assert function_call["function"]["arguments"] == '{"foo":"bar"}'


def test_responses_accumulator_collects_reasoning_summary() -> None:
    """Ensure ResponsesAccumulator captures reasoning summary events."""
    accumulator = ResponsesAccumulator()

    # Add reasoning item start
    reasoning_added = openai_models.ResponseOutputItemAddedEvent(
        sequence_number=1,
        type="response.output_item.added",
        output_index=0,
        item=openai_models.OutputItem(
            id="reason_1",
            type="reasoning",
            status="in_progress",
            summary=[],
        ),
    )
    accumulator.accumulate(reasoning_added.type, reasoning_added.model_dump())

    # Add reasoning summary part and deltas
    summary_added = openai_models.ReasoningSummaryPartAddedEvent(
        sequence_number=2,
        type="response.reasoning_summary_part.added",
        item_id="reason_1",
        output_index=0,
        summary_index=0,
        part=openai_models.ReasoningSummaryPart(type="summary_text", text=""),
    )
    accumulator.accumulate(summary_added.type, summary_added.model_dump())

    summary_delta_one = openai_models.ReasoningSummaryTextDeltaEvent(
        sequence_number=3,
        type="response.reasoning_summary_text.delta",
        item_id="reason_1",
        output_index=0,
        summary_index=0,
        delta="Thought",
    )
    accumulator.accumulate(summary_delta_one.type, summary_delta_one.model_dump())

    summary_delta_two = openai_models.ReasoningSummaryTextDeltaEvent(
        sequence_number=4,
        type="response.reasoning_summary_text.delta",
        item_id="reason_1",
        output_index=0,
        summary_index=0,
        delta=" process",
    )
    accumulator.accumulate(summary_delta_two.type, summary_delta_two.model_dump())

    summary_text_done = openai_models.ReasoningSummaryTextDoneEvent(
        sequence_number=5,
        type="response.reasoning_summary_text.done",
        item_id="reason_1",
        output_index=0,
        summary_index=0,
        text="Thought process",
    )
    accumulator.accumulate(summary_text_done.type, summary_text_done.model_dump())

    summary_part_done = openai_models.ReasoningSummaryPartDoneEvent(
        sequence_number=6,
        type="response.reasoning_summary_part.done",
        item_id="reason_1",
        output_index=0,
        summary_index=0,
        part=openai_models.ReasoningSummaryPart(
            type="summary_text", text="Thought process"
        ),
    )
    accumulator.accumulate(summary_part_done.type, summary_part_done.model_dump())

    # Complete reasoning item with final payload
    reasoning_done = openai_models.ResponseOutputItemDoneEvent(
        sequence_number=7,
        type="response.output_item.done",
        output_index=0,
        item=openai_models.OutputItem(
            id="reason_1",
            type="reasoning",
            status="completed",
            summary=[{"type": "summary_text", "text": "Thought process"}],
        ),
    )
    accumulator.accumulate(reasoning_done.type, reasoning_done.model_dump())

    rebuilt = accumulator.rebuild_response_object(
        {
            "id": "resp_reason",
            "model": "gpt-5",
            "created_at": 0,
            "status": "in_progress",
            "output": {"id": "out_reason", "type": "responses"},
        }
    )

    output_items = rebuilt["output"]
    assert isinstance(output_items, list)
    reasoning_entries = [
        item for item in output_items if item.get("type") == "reasoning"
    ]
    assert reasoning_entries
    entry = reasoning_entries[0]
    assert entry["id"] == "reason_1"
    assert entry["status"] == "completed"
    summary = entry.get("summary")
    assert isinstance(summary, list)
    summary_part = summary[0]
    assert summary_part["text"] == "Thought process"

    reasoning = rebuilt.get("reasoning")
    assert reasoning
    assert isinstance(reasoning.get("summary"), list)
    top_summary = reasoning["summary"][0]
    assert top_summary["text"] == "Thought process"


def test_responses_accumulator_uses_completed_response_payload() -> None:
    """ResponsesAccumulator should leverage the emitted completed response payload."""

    accumulator = ResponsesAccumulator()

    text_added = openai_models.ResponseOutputItemAddedEvent(
        sequence_number=1,
        type="response.output_item.added",
        output_index=0,
        item=openai_models.OutputItem(
            id="text_1",
            type="text",
            text="partial",
            status="in_progress",
        ),
    )
    accumulator.accumulate(text_added.type, text_added.model_dump())

    text_done = openai_models.ResponseOutputItemDoneEvent(
        sequence_number=2,
        type="response.output_item.done",
        output_index=0,
        item=openai_models.OutputItem(
            id="text_1",
            type="text",
            text="Final text",
            status="completed",
        ),
    )
    accumulator.accumulate(text_done.type, text_done.model_dump())

    completed_event = openai_models.ResponseCompletedEvent(
        sequence_number=3,
        type="response.completed",
        response=openai_models.ResponseObject(
            id="resp_completed",
            created_at=123,
            model="gpt-5",
            status="completed",
            output=[
                openai_models.MessageOutput(
                    type="message",
                    id="text_1",
                    status="completed",
                    role="assistant",
                    content=[{"type": "output_text", "text": "Final text"}],
                )
            ],
            usage=openai_models.ResponseUsage(
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
            ),
            parallel_tool_calls=False,
        ),
    )

    accumulator.accumulate(completed_event.type, completed_event.model_dump())

    placeholder_response = {"id": "placeholder", "output": []}
    rebuilt = accumulator.rebuild_response_object(placeholder_response)

    assert rebuilt["id"] == "resp_completed"
    assert rebuilt["status"] == "completed"
    assert isinstance(rebuilt.get("output"), list)
    output_entry = rebuilt["output"][0]
    assert output_entry["content"][0]["text"] == "Final text"
    assert rebuilt["usage"]["total_tokens"] == 3

    completed_copy = accumulator.get_completed_response()
    assert completed_copy == completed_event.response.model_dump()
    if completed_copy is not None:
        completed_copy["status"] = "mutated"
    retrieved_response = accumulator.get_completed_response()
    assert retrieved_response is not None
    assert retrieved_response["status"] == "completed"


def test_responses_accumulator_falls_back_to_output_summary() -> None:
    """ResponsesAccumulator should propagate reasoning summary from output items."""

    accumulator = ResponsesAccumulator()

    completed_event = openai_models.ResponseCompletedEvent(
        sequence_number=1,
        type="response.completed",
        response=openai_models.ResponseObject(
            id="resp_with_summary",
            created_at=123,
            status="completed",
            model="gpt-5",
            output=[
                openai_models.ReasoningOutput(
                    type="reasoning",
                    id="reasoning_1",
                    status="completed",
                    summary=[{"type": "summary_text", "text": "Reasoning trace"}],
                ),
                openai_models.MessageOutput(
                    type="message",
                    id="msg_1",
                    status="completed",
                    role="assistant",
                    content=[{"type": "output_text", "text": "final answer"}],
                ),
            ],
            parallel_tool_calls=False,
        ),
    )

    accumulator.accumulate(completed_event.type, completed_event.model_dump())

    rebuilt = accumulator.rebuild_response_object({})

    reasoning = rebuilt.get("reasoning")
    assert isinstance(reasoning, dict)
    summary = reasoning.get("summary")
    assert isinstance(summary, list)
    assert summary[0]["text"] == "Reasoning trace"


if __name__ == "__main__":
    # Run tests directly
    test_claude_accumulator_rebuild_response()
    test_openai_accumulator_rebuild_response()
    test_responses_accumulator_rebuild_response()
    test_responses_accumulator_collects_reasoning_summary()
    test_responses_accumulator_uses_completed_response_payload()
    test_responses_accumulator_falls_back_to_output_summary()
    print("All tests passed!")
