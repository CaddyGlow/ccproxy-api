"""Behavioral checks for streaming accumulator protocols."""

from __future__ import annotations

from ccproxy.services.adapters.chat_accumulator import ChatCompletionAccumulator


def _chunk_with_tool_call(**overrides: object) -> dict[str, object]:
    base = {
        "id": "chunk",
        "object": "chat.completion.chunk",
        "created": 1696176000,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": '{"param": "value"}'},
                        }
                    ],
                },
            }
        ],
    }
    base.update(overrides)
    return base


def _finish_chunk() -> dict[str, object]:
    return {
        "id": "chunk-finish",
        "object": "chat.completion.chunk",
        "created": 1696176001,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"content": []},
                "finish_reason": "stop",
            }
        ],
    }


def test_chat_accumulator_tracks_tool_calls_until_complete() -> None:
    accumulator = ChatCompletionAccumulator()

    first = _chunk_with_tool_call()
    assert accumulator.accumulate_chunk(first) is None

    second = _chunk_with_tool_call(
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {
                                "name": "test_fn",
                                "arguments": '{"param": "value"}',
                            },
                        }
                    ]
                },
            }
        ],
    )
    assert accumulator.accumulate_chunk(second) is None

    finished = accumulator.accumulate_chunk(_finish_chunk())
    assert finished is not None
    assert finished["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"

    accumulator.reset()

    plain_chunk = {
        "id": "chunk-plain",
        "object": "chat.completion.chunk",
        "created": 1696176002,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"content": [{"type": "text", "text": "hello"}]},
                "finish_reason": None,
            }
        ],
    }

    immediate = accumulator.accumulate_chunk(plain_chunk)
    assert immediate is plain_chunk
