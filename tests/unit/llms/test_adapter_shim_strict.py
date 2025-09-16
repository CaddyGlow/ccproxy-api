from collections.abc import AsyncIterator

import pytest

from ccproxy.llms.adapters.openai_to_openai.chat_to_responses import (
    OpenAIChatToOpenAIResponsesAdapter,
)
from ccproxy.llms.adapters.openai_to_openai.responses_to_chat import (
    OpenAIResponsesToOpenAIChatAdapter,
)
from ccproxy.llms.adapters.shim import AdapterShim


@pytest.mark.asyncio
async def test_shim_adapt_request_chat_to_responses_minimal() -> None:
    shim = AdapterShim(OpenAIChatToOpenAIResponsesAdapter())

    chat_req = {
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "hello world"},
        ],
    }

    out = await shim.adapt_request(chat_req)
    # Expect OpenAI Responses request shape
    assert out.get("model") == "gpt-4"
    assert isinstance(out.get("input"), list)
    assert out["input"][0]["type"] == "message"
    assert out["input"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_shim_adapt_response_responses_to_chat_minimal() -> None:
    shim = AdapterShim(OpenAIChatToOpenAIResponsesAdapter())

    responses_obj = {
        "id": "resp_123",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-4",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "hi"},
                ],
            }
        ],
        "parallel_tool_calls": False,
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }

    out = await shim.adapt_response(responses_obj)
    # Expect Chat Completions result shape
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"]["role"] == "assistant"
    assert "usage" in out


@pytest.mark.asyncio
async def test_shim_adapt_stream_responses_events_to_chat_chunks() -> None:
    shim = AdapterShim(OpenAIResponsesToOpenAIChatAdapter())

    async def gen() -> AsyncIterator[dict]:
        # response.created primes the model id
        yield {
            "sequence_number": 1,
            "type": "response.created",
            "response": {
                "id": "r1",
                "object": "response",
                "created_at": 0,
                "status": "in_progress",
                "model": "gpt-4",
                "output": [],
                "parallel_tool_calls": False,
            },
        }
        # delta text event should produce a ChatCompletionChunk with delta content
        yield {
            "sequence_number": 2,
            "type": "response.output_text.delta",
            "item_id": "it_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hello",
        }
        # completed event should produce a final chunk (finish reason)
        yield {
            "sequence_number": 3,
            "type": "response.completed",
            "response": {
                "id": "r1",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-4",
                "output": [],
                "parallel_tool_calls": False,
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            },
        }

    chunks = []
    async for c in shim.adapt_stream(gen()):
        chunks.append(c)

    assert any(ch["choices"][0]["delta"].get("content") == "hello" for ch in chunks)
    # final chunk should have finish_reason stop
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_shim_adapt_error_normalization() -> None:
    shim = AdapterShim(OpenAIResponsesToOpenAIChatAdapter())

    err = {"error": {"message": "boom", "type": "invalid_request_error"}}
    out = await shim.adapt_error(err)
    assert "error" in out
    assert out["error"]["message"] == "boom"
