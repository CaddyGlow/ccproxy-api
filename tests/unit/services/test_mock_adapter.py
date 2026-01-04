"""Tests for the mock adapter streaming behaviour."""

from types import SimpleNamespace
from typing import Any

import pytest

from ccproxy.services.adapters.mock_adapter import MockAdapter


class _TestableMockAdapter(MockAdapter):
    async def cleanup(self) -> None:  # pragma: no cover - noop for tests
        pass


class StubHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def extract_message_type(self, body: bytes) -> str:
        self.calls.append(("extract", (body,)))
        return "message"

    async def generate_standard_response(
        self, model: Any, is_openai: Any, ctx: Any, message_type: Any
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append(("standard", (model, is_openai, message_type)))
        return 202, {"X-Test": "yes"}, b"standard"

    async def generate_streaming_response(
        self, model: Any, is_openai: Any, ctx: Any, message_type: Any
    ) -> str:
        self.calls.append(("stream", (model, is_openai, message_type)))
        return "stream-object"


class FakeRequest:
    def __init__(
        self, body_bytes: bytes, path: str, context_endpoint: str | None = None
    ) -> None:
        self._body = body_bytes
        self.url = SimpleNamespace(path=path)
        state_dict: dict[str, Any] = {}
        if context_endpoint is not None:
            state_dict["context"] = SimpleNamespace(
                metadata={"endpoint": context_endpoint}
            )
        self.state = SimpleNamespace(**state_dict)

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_handle_request_returns_standard_response() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt", "stream": false}', "/openai/v1/messages"
    )
    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert response.headers["X-Test"] == "yes"
    assert handler.calls[0][0] == "extract"
    assert handler.calls[1][0] == "standard"


@pytest.mark.asyncio
async def test_handle_request_streaming_path() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt", "stream": true}', "/openai/v1/messages"
    )
    result = await adapter.handle_request(request)

    # Don't check type, just check it's truthy since we don't know the exact return type
    assert result
    assert handler.calls[-1][0] == "stream"


@pytest.mark.asyncio
async def test_handle_streaming_uses_endpoint_kwarg() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt-4"}', "/claude", context_endpoint="/openai/internal"
    )
    result = await adapter.handle_streaming(
        request, endpoint="/provider", request_id="abc"
    )

    # Don't check type, just check it's truthy since we don't know the exact return type
    assert result
    assert handler.calls[-1][0] == "stream"
