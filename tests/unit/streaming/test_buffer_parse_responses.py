"""Regression tests for StreamingBufferService._parse_collected_stream.

Covers issue #55: when the Codex Responses API SSE stream has drifted away
from the shape ResponsesAccumulator knows about, the buffer must still emit
a dict whose ``output`` field is a list so the downstream format chain's
ResponseObject validation does not fail with ``Field required``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import ResponsesAccumulator
from ccproxy.streaming.buffer import StreamingBufferService


class _Ctx:
    request_id = "test-req"
    _tool_accumulator_class = ResponsesAccumulator


def _sse(event_type: str, payload: dict[str, Any]) -> bytes:
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body)}\n\n".encode()


@pytest.fixture
def buffer() -> StreamingBufferService:
    return StreamingBufferService(http_client=httpx.AsyncClient())


@pytest.mark.asyncio
async def test_parse_collected_stream_output_is_always_a_list(
    buffer: StreamingBufferService,
) -> None:
    """A Codex stream whose completed event is unrecognizable must still
    yield a dict with ``output`` as a list (issue #55)."""

    base_response = {
        "id": "resp_abc",
        "object": "response",
        "model": "gpt-5-codex",
        "parallel_tool_calls": True,
        "top_logprobs": 0,
    }
    chunks = [
        _sse(
            "response.created",
            {"sequence_number": 1, "response": base_response},
        ),
        _sse(
            "response.in_progress",
            {"sequence_number": 2, "response": base_response},
        ),
        _sse(
            "response.completed",
            {
                "sequence_number": 3,
                "response": {**base_response, "unexpected_future_field": True},
            },
        ),
    ]

    parsed = await buffer._parse_collected_stream(
        chunks=chunks,
        handler_config=None,  # type: ignore[arg-type]
        request_context=_Ctx(),  # type: ignore[arg-type]
    )

    assert parsed is not None
    assert isinstance(parsed.get("output"), list), (
        f"output must be list for ResponseObject validation, got {type(parsed.get('output'))}"
    )
    openai_models.ResponseObject.model_validate(parsed)


@pytest.mark.asyncio
async def test_parse_collected_stream_coerces_non_list_output(
    buffer: StreamingBufferService,
) -> None:
    """Even if upstream sends ``output`` as a bare dict, the buffer coerces it."""

    base_response = {
        "id": "resp_xyz",
        "object": "response",
        "model": "gpt-5-codex",
        "parallel_tool_calls": False,
        "output": {},
    }
    chunks = [
        _sse(
            "response.created",
            {"sequence_number": 1, "response": base_response},
        ),
    ]

    parsed = await buffer._parse_collected_stream(
        chunks=chunks,
        handler_config=None,  # type: ignore[arg-type]
        request_context=_Ctx(),  # type: ignore[arg-type]
    )

    assert parsed is not None
    assert isinstance(parsed.get("output"), list)
    openai_models.ResponseObject.model_validate(parsed)


@pytest.mark.asyncio
async def test_parse_collected_stream_preserves_rebuilt_output(
    buffer: StreamingBufferService,
) -> None:
    """When the accumulator successfully rebuilds message outputs from
    valid events, those outputs must reach the parsed payload."""

    response_dict: dict[str, Any] = {
        "id": "resp_done",
        "object": "response",
        "model": "gpt-5-codex",
        "parallel_tool_calls": False,
        "output": [],
    }
    chunks = [
        _sse(
            "response.created",
            {"sequence_number": 1, "response": response_dict},
        ),
        _sse(
            "response.output_item.added",
            {
                "sequence_number": 2,
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            },
        ),
        _sse(
            "response.output_text.delta",
            {
                "sequence_number": 3,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "hello",
            },
        ),
        _sse(
            "response.output_text.done",
            {
                "sequence_number": 4,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": "hello",
            },
        ),
        _sse(
            "response.output_item.done",
            {
                "sequence_number": 5,
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            },
        ),
        _sse(
            "response.completed",
            {
                "sequence_number": 6,
                "response": {
                    **response_dict,
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_1",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        }
                    ],
                },
            },
        ),
    ]

    parsed = await buffer._parse_collected_stream(
        chunks=chunks,
        handler_config=None,  # type: ignore[arg-type]
        request_context=_Ctx(),  # type: ignore[arg-type]
    )

    assert parsed is not None
    output = parsed.get("output")
    assert isinstance(output, list) and output, (
        "output should contain the rebuilt message"
    )
    validated = openai_models.ResponseObject.model_validate(parsed)
    assert validated.output
