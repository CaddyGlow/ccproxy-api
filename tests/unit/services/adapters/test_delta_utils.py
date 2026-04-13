from __future__ import annotations

from typing import Any

from ccproxy.services.adapters.delta_utils import accumulate_delta


def test_tool_call_type_is_not_concatenated() -> None:
    """Repeated tool_call deltas with type='function' must not concatenate.

    Codex Responses->Chat streaming emits many function_call_arguments.delta
    events, each carrying type='function'. If these strings are merged via
    the generic string-concat branch they collapse into 'functionfunction...'
    which then fails ChatCompletionChunk validation.
    """

    first = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "shell", "arguments": "{"},
                        }
                    ]
                },
            }
        ]
    }
    second = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "function": {"arguments": '"cmd":"ls"}'},
                        }
                    ]
                },
            }
        ]
    }

    merged = accumulate_delta(first, second)
    tool_call = merged["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["index"] == 0
    assert tool_call["function"]["arguments"] == '{"cmd":"ls"}'


def test_tool_call_id_and_name_are_not_concatenated() -> None:
    """Tool-call id/name/call_id must not concatenate across chunks.

    The Codex Responses->Chat stream converter re-emits id/name on every
    function_call_arguments delta chunk. The previous behaviour concatenated
    those strings, producing a 1431-char id and 'shell' x N for the name.
    """

    def make_chunk(arg: str) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": arg,
                                },
                            }
                        ]
                    },
                }
            ]
        }

    acc = make_chunk("{")
    for arg in ('"cmd"', ':"ls"', "}"):
        acc = accumulate_delta(acc, make_chunk(arg))

    tool_call = acc["choices"][0]["delta"]["tool_calls"][0]
    assert tool_call["id"] == "call_abc"
    assert tool_call["function"]["name"] == "shell"
    assert tool_call["function"]["arguments"] == '{"cmd":"ls"}'


def test_call_id_is_not_concatenated() -> None:
    """Responses-style function_call items carry call_id; must not concat."""

    first = {"call_id": "call_xyz", "arguments": "{"}
    second = {"call_id": "call_xyz", "arguments": '"a":1}'}
    merged = accumulate_delta(first, second)
    assert merged["call_id"] == "call_xyz"
    assert merged["arguments"] == '{"a":1}'
