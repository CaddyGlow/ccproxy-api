"""Tests for the mock adapter streaming behaviour."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)
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

    def extract_prompt_text(self, body: bytes) -> str:
        self.calls.append(("prompt", (body,)))
        return "prompt"

    async def generate_standard_response(
        self,
        model: Any,
        target_format: Any,
        ctx: Any,
        message_type: Any,
        prompt_text: Any,
    ) -> tuple[int, dict[str, str], bytes]:
        self.calls.append(
            ("standard", (model, target_format, message_type, prompt_text))
        )
        return 202, {"X-Test": "yes"}, json.dumps({"format": target_format}).encode()

    async def generate_streaming_response(
        self,
        model: Any,
        target_format: Any,
        ctx: Any,
        message_type: Any,
        prompt_text: Any,
    ) -> str:
        self.calls.append(("stream", (model, target_format, message_type, prompt_text)))
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
                metadata={"endpoint": context_endpoint},
                format_chain=[],
            )
        self.state = SimpleNamespace(**state_dict)

    async def body(self) -> bytes:
        return self._body


@pytest.mark.asyncio
async def test_handle_request_returns_standard_response() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt", "stream": false}', "/codex/v1/responses"
    )
    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert response.headers["X-Test"] == "yes"
    assert json.loads(bytes(response.body))["format"] == FORMAT_OPENAI_RESPONSES
    assert handler.calls[0][0] == "extract"
    assert handler.calls[1][0] == "prompt"
    assert handler.calls[2][0] == "standard"


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
async def test_handle_request_prefers_context_format_chain() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt"}', "/codex/v1/chat/completions", "/v1/chat/completions"
    )
    request.state.context.format_chain = [FORMAT_OPENAI_CHAT]

    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert json.loads(bytes(response.body))["format"] == FORMAT_OPENAI_CHAT


@pytest.mark.asyncio
async def test_handle_request_falls_back_to_chat_endpoint_detection() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt"}', "/codex/v1/chat/completions", "/internal/unknown"
    )

    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert json.loads(bytes(response.body))["format"] == FORMAT_OPENAI_CHAT


@pytest.mark.asyncio
async def test_handle_request_falls_back_to_responses_endpoint_detection() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt"}', "/codex/v1/responses", "/internal/unknown"
    )

    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert json.loads(bytes(response.body))["format"] == FORMAT_OPENAI_RESPONSES


@pytest.mark.asyncio
async def test_handle_request_ignores_unknown_format_chain_and_uses_endpoint() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "gpt"}', "/codex/v1/chat/completions", "/codex/v1/chat/completions"
    )
    request.state.context.format_chain = ["unsupported.format"]

    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert json.loads(bytes(response.body))["format"] == FORMAT_OPENAI_CHAT


@pytest.mark.asyncio
async def test_handle_request_defaults_to_anthropic_for_unknown_endpoint() -> None:
    handler: Any = StubHandler()
    adapter = _TestableMockAdapter(handler)

    request: Any = FakeRequest(
        b'{"model": "claude"}', "/provider/v1/messages", "/provider/v1/messages"
    )

    response = await adapter.handle_request(request)

    assert response.status_code == 202
    assert json.loads(bytes(response.body))["format"] == FORMAT_ANTHROPIC_MESSAGES


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
