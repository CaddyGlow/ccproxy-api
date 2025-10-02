"""External Codex API mocks driven by recorded samples."""

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


def _resolve_codex_sample(payload: dict[str, Any]) -> str:
    suffix = "responses"

    has_tool_payload = any(
        bool(payload.get(key)) for key in ("tools", "tool_results", "tool_outputs")
    )

    if has_tool_payload:
        suffix += "_tools"
    elif payload.get("response_format") or payload.get("text"):
        suffix += "_structured"
    elif payload.get("reasoning"):
        suffix += "_thinking"

    if payload.get("stream"):
        suffix += "_stream"

    return f"codex_{suffix}"


@pytest.fixture
def mock_external_codex_api(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Intercept Codex upstream calls and respond with recorded samples."""

    # Note: HTTPXMock automatically allows reusing callbacks via is_reusable=True
    # No need to set can_send_already_matched_responses

    def _callback(request: httpx.Request) -> httpx.Response:
        print(
            f"🌐 [MOCK] Intercepted upstream Codex API request: {request.method} {request.url}"
        )
        try:
            payload = json.loads(request.content.decode() or "{}")
        except json.JSONDecodeError:
            payload = {}

        sample_name = _resolve_codex_sample(payload)

        sample = _SAMPLE_CACHE.get(sample_name)
        if sample is None:
            sample = load_sample(sample_name)
            _SAMPLE_CACHE[sample_name] = sample

        status_code, headers, content = response_content_from_sample(sample)
        print(
            f"🌐 [MOCK] Returning mocked Codex response: {status_code}, sample={sample_name}"
        )
        return httpx.Response(status_code=status_code, headers=headers, content=content)

    httpx_mock.add_callback(
        _callback,
        url="https://chatgpt.com/backend-api/codex/responses",
        is_reusable=True,
    )
    return httpx_mock
