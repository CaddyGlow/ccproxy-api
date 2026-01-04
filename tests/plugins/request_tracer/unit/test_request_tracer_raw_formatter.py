from __future__ import annotations

from pathlib import Path

import pytest

from ccproxy.core.plugins.hooks.implementations.formatters.raw import RawHTTPFormatter
from ccproxy.plugins.request_tracer.config import RequestTracerConfig


@pytest.mark.asyncio
async def test_raw_formatter_writes_files(tmp_path: Path) -> None:
    cfg = RequestTracerConfig(raw_http_enabled=True, raw_log_dir=str(tmp_path))
    fmt = RawHTTPFormatter.from_config(cfg)

    assert fmt.should_log() is True

    req_id = "abc123"
    await fmt.log_client_request(req_id, b"GET / HTTP/1.1\r\n\r\n")
    await fmt.log_client_response(req_id, b"HTTP/1.1 200 OK\r\n\r\n")
    await fmt.log_provider_request(req_id, b"POST /v1/messages HTTP/1.1\r\n\r\n")
    await fmt.log_provider_response(req_id, b"HTTP/1.1 200 OK\r\n\r\n")

    # Ensure files exist (with timestamp-based names)
    files = list(tmp_path.glob("*.http"))
    request_files = [f for f in files if "client_request" in f.name]
    response_files = [f for f in files if "client_response" in f.name]
    provider_request_files = [f for f in files if "provider_request" in f.name]
    provider_response_files = [f for f in files if "provider_response" in f.name]

    assert len(request_files) == 1
    assert len(response_files) == 1
    assert len(provider_request_files) == 1
    assert len(provider_response_files) == 1


@pytest.mark.asyncio
async def test_raw_formatter_respects_size_limit(tmp_path: Path) -> None:
    cfg = RequestTracerConfig(
        raw_http_enabled=True, raw_log_dir=str(tmp_path), max_body_size=5
    )
    fmt = RawHTTPFormatter.from_config(cfg)

    body = b"0123456789"
    await fmt.log_client_request("rid", body)

    # Find the generated file
    files = list(tmp_path.glob("*_client_request*.http"))
    assert len(files) == 1
    content = files[0].read_bytes()
    # Expect truncation marker
    assert content.endswith(b"[TRUNCATED]")
