import pytest


@pytest.mark.asyncio
async def test_openai_responses_stream_function_call_arguments_to_anthropic_tool_use():
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
async def test_openai_responses_stream_reasoning_summary_to_anthropic_thinking():
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
