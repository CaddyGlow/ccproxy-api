"""Tests for provider request hook integration in BaseHTTPAdapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.core.plugins.hooks.manager import HookManager
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter


class _DummyPool:
    """Minimal HTTP pool stub exposing the hook manager."""

    def __init__(self, hook_manager: HookManager | None) -> None:
        self.hook_manager = hook_manager

    async def get_client(self, *args: Any, **kwargs: Any) -> object:
        return object()


class _DummyStreamingHandler:
    """Streaming handler stub that records invocation arguments."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] | None = None

    async def handle_streaming_request(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        handler_config: Any,
        request_context: Any,
        client: Any,
    ) -> StreamingResponse:
        self.captured = {
            "method": method,
            "url": url,
            "headers": headers,
            "body": body,
        }
        return StreamingResponse(iter([b"chunk"]))


class _TestHTTPAdapter(BaseHTTPAdapter):
    """Concrete adapter for exercising hook behaviour in tests."""

    def __init__(
        self,
        *,
        hook_manager: HookManager | None,
        streaming_handler: _DummyStreamingHandler | None = None,
    ) -> None:
        config = ProviderConfig(
            name="test-provider", base_url="https://api.example.com"
        )
        super().__init__(
            config=config,
            auth_manager=None,
            http_pool_manager=_DummyPool(hook_manager),
            streaming_handler=streaming_handler,
        )
        self.captured_request: tuple[str, str, dict[str, str], bytes] | None = None

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        merged = dict(headers)
        merged.setdefault("content-type", "application/json")
        return body, merged

    async def process_provider_response(self, response: Any, endpoint: str) -> Response:
        return Response(content=response.content, status_code=response.status_code)

    async def get_target_url(self, endpoint: str) -> str:
        suffix = endpoint or "default"
        return f"{self.config.base_url}/{suffix}"

    async def _execute_http_request(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> Any:
        self.captured_request = (method, url, headers, body)

        class _FakeResponse:
            status_code = 200
            headers = {"content-type": "application/json"}
            content = b"{}"

        return _FakeResponse()


def _make_request(body: bytes, headers: dict[str, str], ctx: Any) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/test",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    called = False

    async def receive() -> dict[str, Any]:
        nonlocal called
        if not called:
            called = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    request.state.context = ctx
    return request


class _MutateHook:
    name = "mutate-provider-request"
    events = [HookEvent.PROVIDER_REQUEST_PREPARED]
    priority = 100

    async def __call__(
        self, context: HookContext
    ) -> None:  # pragma: no cover - tested indirectly
        assert context.metadata.get("request_id") is not None
        assert context.provider == "test-provider"

        payload = context.data.get("body")
        if isinstance(payload, dict):
            payload["mutated"] = True
            context.data["body"] = payload

        headers = dict(context.data.get("headers", {}))
        headers["x-hook"] = "1"
        context.data["headers"] = headers

        url = context.data.get("url", "")
        context.data["url"] = f"{url}?via=hook"


class _RawOverrideHook:
    name = "raw-body-override"
    events = [HookEvent.PROVIDER_REQUEST_PREPARED]
    priority = 100

    async def __call__(
        self, context: HookContext
    ) -> None:  # pragma: no cover - tested indirectly
        context.data["body_raw"] = b"override-body"


def _make_hook_manager(*hooks: Any) -> HookManager:
    registry = HookRegistry()
    for hook in hooks:
        registry.register(hook)
    return HookManager(registry)


@pytest.mark.asyncio
async def test_provider_request_prepared_hook_mutates_outbound_request() -> None:
    hook_manager = _make_hook_manager(_MutateHook())
    adapter = _TestHTTPAdapter(hook_manager=hook_manager)

    ctx = SimpleNamespace(
        metadata={"endpoint": "chat"}, format_chain=[], request_id="req-123"
    )
    request = _make_request(
        b'{"model":"foo"}', {"content-type": "application/json"}, ctx
    )

    response = await adapter.handle_request(request)
    assert response.status_code == 200

    assert adapter.captured_request is not None
    method, url, headers, body = adapter.captured_request
    assert method == "POST"
    assert url.endswith("?via=hook")
    assert headers["x-hook"] == "1"

    payload = json.loads(body.decode())
    assert payload["mutated"] is True


@pytest.mark.asyncio
async def test_provider_request_prepared_hook_can_override_raw_body() -> None:
    hook_manager = _make_hook_manager(_RawOverrideHook())
    adapter = _TestHTTPAdapter(hook_manager=hook_manager)

    ctx = SimpleNamespace(
        metadata={"endpoint": "chat"}, format_chain=[], request_id="req-456"
    )
    request = _make_request(
        b'{"model":"foo"}', {"content-type": "application/json"}, ctx
    )

    await adapter.handle_request(request)

    assert adapter.captured_request is not None
    _, _, _, body = adapter.captured_request
    assert body == b"override-body"


@pytest.mark.asyncio
async def test_provider_request_prepared_hook_applies_to_streaming_requests() -> None:
    hook_manager = _make_hook_manager(_MutateHook())
    streaming_handler = _DummyStreamingHandler()
    adapter = _TestHTTPAdapter(
        hook_manager=hook_manager, streaming_handler=streaming_handler
    )

    ctx = SimpleNamespace(
        metadata={"endpoint": "chat"}, format_chain=[], request_id="req-stream"
    )
    request = _make_request(
        b'{"model":"foo"}', {"content-type": "application/json"}, ctx
    )

    await adapter.handle_streaming(request, endpoint="chat")

    assert streaming_handler.captured is not None
    captured = streaming_handler.captured
    assert captured["method"] == "POST"
    assert captured["url"].endswith("?via=hook")
    assert captured["headers"]["x-hook"] == "1"

    payload = json.loads(captured["body"].decode())
    assert payload["mutated"] is True
