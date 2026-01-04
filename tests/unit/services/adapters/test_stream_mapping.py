from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import ValidationError

from ccproxy.llms.models.openai import ChatCompletionChunk
from ccproxy.services.adapters.simple_converters import map_stream


async def _aiter(items: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    for it in items:
        yield it


class DummyConverter:
    def __init__(self) -> None:
        self.calls: int = 0

    def __call__(self, stream: AsyncIterator[Any]) -> AsyncIterator[Any]:  # type: ignore[override]
        async def _gen() -> AsyncIterator[Any]:
            async for evt in stream:
                # echo back a minimal object with model_dump method
                self.calls += 1

                # Build a plain dict from evt
                if isinstance(evt, dict):
                    data = evt
                elif hasattr(evt, "model_dump"):
                    data = evt.model_dump(exclude_unset=True)  # type: ignore[attr-defined]
                else:
                    data = dict(getattr(evt, "__dict__", {}))

                class Obj:
                    def __init__(self, d: dict[str, Any]) -> None:
                        self._d = d

                    def model_dump(
                        self, *, exclude_unset: bool = True
                    ) -> dict[str, Any]:
                        return self._d

                yield Obj(data)

        return _gen()


@pytest.mark.asyncio
async def test_map_stream_validates_and_maps() -> None:
    # Provide minimal-but-valid chunks so the validator succeeds without fallbacks
    chunks: list[dict[str, Any]] = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "c2",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "world"},
                    "finish_reason": None,
                }
            ],
        },
    ]

    dummy = DummyConverter()

    out: list[dict[str, Any]] = []
    async for item in map_stream(
        _aiter(chunks), validator_model=ChatCompletionChunk, converter=dummy
    ):
        out.append(item)

    assert len(out) == 2
    assert out[0]["id"] == "c1"
    assert out[1]["id"] == "c2"
    # Converter sees two events
    assert dummy.calls == 2


@pytest.mark.asyncio
async def test_map_stream_raises_on_invalid_data() -> None:
    # Invalid payloads now bubble up as validation errors without dict fallbacks
    chunks = [
        {"unexpected": True},
    ]
    dummy = DummyConverter()

    with pytest.raises(ValidationError):
        async for _ in map_stream(
            _aiter(chunks), validator_model=ChatCompletionChunk, converter=dummy
        ):
            pass

    assert dummy.calls == 0


@pytest.mark.asyncio
async def test_map_stream_valid_tool_call_succeeds() -> None:
    """Test that properly formed tool call data validates successfully."""
    valid_tool_call_chunk = {
        "id": "test-chunk-id",
        "object": "chat.completion.chunk",
        "created": 1696176000,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call_123",  # Required field present
                            "function": {
                                "name": "test_function",  # Required field present
                                "arguments": '{"param": "value"}',  # Valid JSON
                            },
                            "index": 0,
                        }
                    ]
                },
            }
        ],
    }

    dummy = DummyConverter()

    out: list[dict[str, Any]] = []
    # This should succeed without ValidationError
    async for item in map_stream(
        _aiter([valid_tool_call_chunk]),
        validator_model=ChatCompletionChunk,
        converter=dummy,
    ):
        out.append(item)

    assert len(out) == 1
    assert out[0]["id"] == "test-chunk-id"
    assert dummy.calls == 1


@pytest.mark.asyncio
async def test_map_stream_with_accumulator() -> None:
    """Test that map_stream works with an accumulator."""
    from ccproxy.services.adapters.chat_accumulator import ChatCompletionAccumulator

    # Simulate partial tool call chunks that will be accumulated
    partial_chunks = [
        # First chunk: tool call start with index but no id
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1696176000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"param"'}}
                        ]
                    },
                }
            ],
        },
        # Second chunk: tool call id and name
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 1696176000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": ': "value"}',
                                },
                            }
                        ]
                    },
                }
            ],
        },
        # Third chunk: finish reason
        {
            "id": "chunk-3",
            "object": "chat.completion.chunk",
            "created": 1696176000,
            "model": "gpt-4",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]

    accumulator = ChatCompletionAccumulator()
    dummy = DummyConverter()

    out: list[dict[str, Any]] = []
    # This should succeed by accumulating partial chunks
    async for item in map_stream(
        _aiter(partial_chunks),
        validator_model=ChatCompletionChunk,
        converter=dummy,
        accumulator=accumulator,
    ):
        out.append(item)

    # Should get one accumulated complete chunk
    assert len(out) == 1

    # Verify the accumulated tool call is complete
    accumulated_chunk = out[0]
    tool_calls = accumulated_chunk["choices"][0]["delta"]["tool_calls"]
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["id"] == "call_123"
    assert tool_call["function"]["name"] == "test_function"
    assert tool_call["function"]["arguments"] == '{"param": "value"}'

    # Converter should be called once for the complete chunk
    assert dummy.calls == 1


@pytest.mark.asyncio
async def test_map_stream_accumulator_handles_regular_chunks() -> None:
    """Test that accumulator passes through regular chunks immediately."""
    from ccproxy.services.adapters.chat_accumulator import ChatCompletionAccumulator

    # Regular content chunks (no tool calls)
    regular_chunks = [
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 1696176000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
            ],
        },
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 1696176000,
            "model": "gpt-4",
            "choices": [
                {"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}
            ],
        },
    ]

    accumulator = ChatCompletionAccumulator()
    dummy = DummyConverter()

    out: list[dict[str, Any]] = []
    async for item in map_stream(
        _aiter(regular_chunks),
        validator_model=ChatCompletionChunk,
        converter=dummy,
        accumulator=accumulator,
    ):
        out.append(item)

    # Should get both chunks processed immediately
    assert len(out) == 2
    assert out[0]["choices"][0]["delta"]["content"] == "Hello"
    assert out[1]["choices"][0]["delta"]["content"] == " world"
    assert dummy.calls == 2
