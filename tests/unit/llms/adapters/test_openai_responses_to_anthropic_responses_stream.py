import pytest


@pytest.mark.asyncio
async def test_openai_responses_stream_function_call_arguments_to_anthropic_tool_use():
    from ccproxy.llms.adapters.openai_responses_to_anthropic_responses import (
        OpenAIResponsesToAnthropicAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}
        yield {"type": "response.function_call_arguments.delta", "item_id": "it1", "output_index": 0, "delta": '{"q":'}
        yield {"type": "response.function_call_arguments.delta", "item_id": "it1", "output_index": 0, "delta": '"abc"}'}
        yield {"type": "response.function_call_arguments.done", "item_id": "it1", "output_index": 0, "arguments": '{"q":"abc"}'}
        yield {"type": "response.completed", "response": {}}

    adapter = OpenAIResponsesToAnthropicAdapter()
    out = []
    async for ev in adapter.adapt_stream(gen()):
        out.append(ev)

    assert any(e.get("type") == "content_block_start" and (e.get("content_block") or {}).get("type") == "tool_use" for e in out)
    assert any(e.get("type") == "content_block_stop" for e in out)
