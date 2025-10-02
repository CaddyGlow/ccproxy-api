"""Unit tests for CORS handling in the endpoint testing harness."""

from __future__ import annotations

import httpx
import pytest

from ccproxy.testing.endpoints.runner import TestEndpoint


TestEndpoint.__test__ = False  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_post_json_includes_origin_and_default_headers() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = {k.lower(): v for k, v in request.headers.items()}
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tester = TestEndpoint(
        base_url="http://proxy.local",
        cors_origin="https://ui.example",
        default_headers={"X-Env": "test"},
        client=client,
    )

    async with tester:
        response_body = await tester.post_json(
            "http://proxy.local/chat/completions",
            {"model": "gpt", "messages": []},
            headers={"X-Request": "cors"},
        )

    assert response_body == {"ok": True}
    assert captured["origin"] == "https://ui.example"
    assert captured["content-type"] == "application/json"
    assert captured["accept-encoding"] == "identity"
    assert captured["x-env"] == "test"
    assert captured["x-request"] == "cors"


@pytest.mark.asyncio
async def test_options_preflight_sets_cors_headers() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = {k.lower(): v for k, v in request.headers.items()}
        return httpx.Response(
            204,
            headers={
                "Access-Control-Allow-Origin": "https://ui.example",
                "Access-Control-Allow-Methods": "POST",
                "Access-Control-Allow-Headers": "Authorization",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tester = TestEndpoint(
        base_url="http://proxy.local",
        cors_origin="https://ui.example",
        client=client,
    )

    async with tester:
        status_code, response_headers = await tester.options_preflight(
            "http://proxy.local/chat/completions",
            request_method="POST",
            request_headers=["Authorization", "Content-Type"],
        )

    assert status_code == 204
    assert captured["origin"] == "https://ui.example"
    assert captured["access-control-request-method"] == "POST"
    assert captured["access-control-request-headers"] == "Authorization, Content-Type"
    assert response_headers.get("access-control-allow-origin") == "https://ui.example"
    assert response_headers.get("access-control-allow-methods") == "POST"
    assert response_headers.get("access-control-allow-headers") == "Authorization"
