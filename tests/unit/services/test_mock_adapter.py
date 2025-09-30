"""Tests for the mock adapter streaming behaviour."""

from types import SimpleNamespace

import pytest

from ccproxy.services.adapters.mock_adapter import MockAdapter


class _TestableMockAdapter(MockAdapter):
    async def cleanup(self) -> None:  # pragma: no cover - noop for tests
        pass


class StubHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def extract_message_type(self, body: bytes) -> str:
        self.calls.append(("extract", (body,)))
        return "message"

    async def generate_standard_response(self, model, is_openai, ctx, message_type):
        self.calls.append(("standard", (model, is_openai, message_type)))
        return 202, {"X-Test": "yes"}, b"standard"

    async def generate_streaming_response(self, model, is_openai, ctx, message_type):
        self.calls.append(("stream", (model, is_openai, message_type)))
        return "stream-object"


class FakeRequest:
    def __init__(
        self, body_bytes: bytes, path: str, context_endpoint: str | None = None
    ) -> None:
        self._body = body_bytes
        self.url = SimpleNamespace(path=path)
        state_dict = {}
        if context_endpoint is not None:
            state_dict["context"] = SimpleNamespace(
                metadata={"endpoint": context_endpoint}
            )
        self.state = SimpleNamespace(**state_dict)

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_handle_request_returns_standard_response():
    handler = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request = FakeRequest(b'{"model": "gpt", "stream": false}', "/openai/v1/messages")
    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert response.headers["X-Test"] == "yes"
    assert handler.calls[0][0] == "extract"
    assert handler.calls[1][0] == "standard"


@pytest.mark.asyncio
async def test_handle_request_streaming_path():
    handler = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request = FakeRequest(b'{"model": "gpt", "stream": true}', "/openai/v1/messages")
    result = await adapter.handle_request(request)

    assert result == "stream-object"
    assert handler.calls[-1][0] == "stream"


@pytest.mark.asyncio
async def test_handle_streaming_uses_endpoint_kwarg():
    handler = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request = FakeRequest(
        b'{"model": "gpt-4"}', "/claude", context_endpoint="/openai/internal"
    )
    result = await adapter.handle_streaming(
        request, endpoint="/provider", request_id="abc"
    )

    assert result == "stream-object"
    assert handler.calls[-1][0] == "stream"
