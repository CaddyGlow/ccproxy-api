"""External Copilot API mocks driven by recorded samples."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock
from tests.helpers.sample_loader import (
    load_sample,
    response_content_from_sample,
)


_SAMPLE_CACHE: dict[str, dict[str, Any]] = {}


def _resolve_copilot_sample(payload: dict[str, Any]) -> str:
    suffix = "chat_completions"

    has_tool_data = any(
        bool(payload.get(key)) for key in ("tools", "tool_results", "tool_outputs")
    )

    if has_tool_data:
        suffix += "_tools"
    elif payload.get("response_format"):
        suffix += "_structured"
    elif payload.get("reasoning"):
        suffix += "_thinking"

    if payload.get("stream"):
        suffix += "_stream"

    return f"copilot_{suffix}"


@pytest.fixture
def mock_external_copilot_api(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Intercept Copilot upstream calls and respond with recorded samples."""

    def _callback(request: httpx.Request) -> httpx.Response:
        print(
            f"🌐 [MOCK] Intercepted upstream Copilot API request: {request.method} {request.url}"
        )
        host = request.url.host or ""
        if "githubcopilot.com" not in host:
            print(f"🌐 [MOCK] Rejected non-Copilot host: {host}")
            return httpx.Response(status_code=404)

        path = request.url.path or ""
        if not (
            path.endswith("/chat/completions") or path.endswith("/v1/chat/completions")
        ):
            print(f"🌐 [MOCK] Rejected non-chat-completions path: {path}")
            return httpx.Response(status_code=404)

        try:
            payload = json.loads(request.content.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}

        sample_name = _resolve_copilot_sample(payload)

        sample = _SAMPLE_CACHE.get(sample_name)
        if sample is None:
            sample = load_sample(sample_name)
            _SAMPLE_CACHE[sample_name] = sample

        status_code, headers, content = response_content_from_sample(sample)
        print(
            f"🌐 [MOCK] Returning mocked Copilot response: {status_code}, sample={sample_name}"
        )
        return httpx.Response(status_code=status_code, headers=headers, content=content)

    # Configure to allow multiple requests to this callback
    httpx_mock.add_callback(_callback, is_reusable=True)
    return httpx_mock
