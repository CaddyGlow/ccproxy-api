from __future__ import annotations

from pathlib import Path

import pytest

from plugins.request_tracer.config import RequestTracerConfig
from plugins.request_tracer.formatters.raw import RawHTTPFormatter


@pytest.mark.asyncio
async def test_raw_formatter_writes_files(tmp_path: Path) -> None:
    cfg = RequestTracerConfig(raw_http_enabled=True, raw_log_dir=str(tmp_path))
    fmt = RawHTTPFormatter(cfg)

    assert fmt.should_log() is True

    req_id = "abc123"
    await fmt.log_client_request(req_id, b"GET / HTTP/1.1\r\n\r\n")
    await fmt.log_client_response(req_id, b"HTTP/1.1 200 OK\r\n\r\n")
    await fmt.log_provider_request(req_id, b"POST /v1/messages HTTP/1.1\r\n\r\n")
    await fmt.log_provider_response(req_id, b"HTTP/1.1 200 OK\r\n\r\n")

    # Ensure files exist
    assert (tmp_path / f"{req_id}_client_request.http").exists()
    assert (tmp_path / f"{req_id}_client_response.http").exists()
    assert (tmp_path / f"{req_id}_provider_request.http").exists()
    assert (tmp_path / f"{req_id}_provider_response.http").exists()


@pytest.mark.asyncio
async def test_raw_formatter_respects_size_limit(tmp_path: Path) -> None:
    cfg = RequestTracerConfig(
        raw_http_enabled=True, raw_log_dir=str(tmp_path), max_body_size=5
    )
    fmt = RawHTTPFormatter(cfg)

    body = b"0123456789"
    await fmt.log_client_request("rid", body)

    content = (tmp_path / "rid_client_request.http").read_bytes()
    # Expect truncation marker
    assert content.endswith(b"[TRUNCATED]")
