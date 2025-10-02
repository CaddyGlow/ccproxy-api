"""Tests for the core HTTP tracer hook."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.core.plugins.hooks.implementations.http_tracer import HTTPTracerHook


class StubJSONFormatter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def log_request(
        self,
        *,
        request_id: str,
        method: str,
        url: str,
        headers: object,
        body: object,
        request_type: str,
        context: object = None,
        hook_type: object = None,
    ) -> None:
        self.calls.append(
            (
                "request",
                {
                    "request_id": request_id,
                    "method": method,
                    "url": url,
                    "request_type": request_type,
                    "body": body,
                    "hook_type": hook_type,
                },
            )
        )

    async def log_response(
        self,
        *,
        request_id: str,
        status: int,
        headers: object,
        body: bytes,
        response_type: str,
        context: object = None,
        hook_type: object = None,
    ) -> None:
        self.calls.append(
            (
                "response",
                {
                    "request_id": request_id,
                    "status": status,
                    "response_type": response_type,
                    "body": body,
                    "hook_type": hook_type,
                },
            )
        )

    async def log_error(
        self, *, request_id: str, error: object, **_kwargs: object
    ) -> None:
        self.calls.append(("error", {"request_id": request_id, "message": str(error)}))


class StubRawFormatter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def log_provider_request(
        self, *, request_id: str, raw_data: bytes, hook_type: object = None
    ) -> None:
        self.calls.append(
            (
                "provider_request",
                {"id": request_id, "hook": hook_type, "data": raw_data},
            )
        )

    async def log_client_request(
        self, *, request_id: str, raw_data: bytes, hook_type: object = None
    ) -> None:
        self.calls.append(
            ("client_request", {"id": request_id, "hook": hook_type, "data": raw_data})
        )

    async def log_provider_response(
        self, *, request_id: str, raw_data: bytes, hook_type: object = None
    ) -> None:
        self.calls.append(
            (
                "provider_response",
                {"id": request_id, "hook": hook_type, "data": raw_data},
            )
        )

    async def log_client_response(
        self, *, request_id: str, raw_data: bytes, hook_type: object = None
    ) -> None:
        self.calls.append(
            ("client_response", {"id": request_id, "hook": hook_type, "data": raw_data})
        )


@pytest.mark.asyncio
async def test_http_tracer_logs_provider_request_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_formatter = StubJSONFormatter()
    raw_formatter = StubRawFormatter()

    monkeypatch.setattr(
        "ccproxy.core.plugins.hooks.implementations.http_tracer.logger",
        SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )

    hook = HTTPTracerHook(json_formatter=json_formatter, raw_formatter=raw_formatter)

    request_context = HookContext(
        event=HookEvent.HTTP_REQUEST,
        timestamp=datetime.utcnow(),
        data={
            "request_id": "req-1",
            "method": "POST",
            "url": "https://api.anthropic.com/v1/messages",
            "headers": {"Content-Type": "application/json"},
            "body": {"hello": "world"},
            "is_json": True,
        },
        metadata={},
    )

    await hook(request_context)

    response_context = HookContext(
        event=HookEvent.HTTP_RESPONSE,
        timestamp=datetime.utcnow(),
        data={
            "request_id": "req-1",
            "status_code": 200,
            "url": "https://api.anthropic.com/v1/messages",
            "response_headers": [("Content-Type", "application/json")],
            "response_body": {"ok": True},
        },
        metadata={},
    )

    await hook(response_context)

    assert json_formatter.calls[0][0] == "request"
    assert json_formatter.calls[0][1]["request_type"] == "provider"
    assert json_formatter.calls[1][0] == "response"
    assert json_formatter.calls[1][1]["response_type"] == "provider"

    assert raw_formatter.calls[0][0] == "provider_request"
    assert raw_formatter.calls[1][0] == "provider_response"


@pytest.mark.asyncio
async def test_http_tracer_logs_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    json_formatter = StubJSONFormatter()
    raw_formatter = StubRawFormatter()

    monkeypatch.setattr(
        "ccproxy.core.plugins.hooks.implementations.http_tracer.logger",
        SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    )

    hook = HTTPTracerHook(json_formatter=json_formatter, raw_formatter=raw_formatter)

    error_context = HookContext(
        event=HookEvent.HTTP_ERROR,
        timestamp=datetime.utcnow(),
        data={
            "request_id": "req-err",
            "url": "https://localhost/api",
            "status_code": 500,
            "error_type": "HTTPException",
            "error_detail": "boom",
            "response_body": "failure",
        },
        metadata={},
    )

    await hook(error_context)

    assert json_formatter.calls[0][0] == "error"
    error_message = json_formatter.calls[0][1]["message"]
    assert isinstance(error_message, str) and "boom" in error_message
    assert raw_formatter.calls[0][0] == "client_response"
