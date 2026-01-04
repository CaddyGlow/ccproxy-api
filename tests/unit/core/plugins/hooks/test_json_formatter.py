"""Tests for the JSONFormatter tracing helper."""

import json
from pathlib import Path

import pytest

from ccproxy.core.plugins.hooks.implementations.formatters.json import JSONFormatter


@pytest.mark.asyncio
async def test_log_request_writes_json(tmp_path: Path) -> None:
    formatter = JSONFormatter(log_dir=str(tmp_path))

    await formatter.log_request(
        request_id="req-1",
        method="POST",
        url="https://example.test",
        headers={"Authorization": "shh", "X-Request": "123"},
        body=b'{"hello": "world"}',
    )

    files = list(tmp_path.glob("*_request.json"))
    assert files, "request log file not written"

    data = json.loads(files[0].read_text())
    assert data["headers"]["Authorization"] == "shh"
    assert data["body"]["hello"] == "world"


@pytest.mark.asyncio
async def test_log_response_writes_plain_body(tmp_path: Path) -> None:
    formatter = JSONFormatter(log_dir=str(tmp_path))

    await formatter.log_response(
        request_id="req-2",
        status=200,
        headers={"Content-Type": "text/plain"},
        body=b"plain text body",
    )

    files = list(tmp_path.glob("*_response.json"))
    assert files, "response log file not written"

    data = json.loads(files[0].read_text())
    assert data["status"] == 200
    assert data["body"] == "plain text body"


def test_redact_headers_masks_sensitive_values() -> None:
    masked = JSONFormatter.redact_headers(
        {
            "Authorization": "secret",
            "X-API-KEY": "abc",
            "X-Custom": "visible",
        }
    )

    assert masked["Authorization"] == "[REDACTED]"
    assert masked["X-API-KEY"] == "[REDACTED]"
    assert masked["X-Custom"] == "visible"


def test_body_preview_formats_json() -> None:
    formatter = JSONFormatter(json_logs_enabled=False)
    preview = formatter._get_body_preview(b'{"foo": "bar"}')

    assert "foo" in preview


@pytest.mark.asyncio
async def test_log_error_runs_without_exception() -> None:
    formatter = JSONFormatter(json_logs_enabled=False)

    await formatter.log_error("req-3", Exception("boom"), duration=0.1, provider="svc")
