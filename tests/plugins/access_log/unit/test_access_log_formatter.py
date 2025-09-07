from __future__ import annotations

import json

from ccproxy.plugins.access_log.formatter import AccessLogFormatter


def sample_data() -> dict[str, object]:
    return {
        "timestamp": 1735867200.0,  # fixed time for predictability
        "request_id": "req-123",
        "method": "GET",
        "path": "/api/v1/foo",
        "query": "a=1&b=2",
        "status_code": 200,
        "duration_ms": 12.5,
        "client_ip": "127.0.0.1",
        "user_agent": "pytest-agent",
        "body_size": 123,
    }


def test_format_common_contains_expected_parts() -> None:
    fmt = AccessLogFormatter()
    line = fmt.format_client(sample_data(), "common")

    assert "127.0.0.1" in line
    assert "GET /api/v1/foo?a=1&b=2 HTTP/1.1" in line
    assert " 200 123" in line


def test_format_combined_includes_user_agent() -> None:
    fmt = AccessLogFormatter()
    line = fmt.format_client(sample_data(), "combined")

    assert '"pytest-agent"' in line
    # Referer is "-" by default
    assert ' "-" ' in line


def test_format_structured_client_is_json() -> None:
    fmt = AccessLogFormatter()
    s = fmt.format_client(sample_data(), "structured")
    data = json.loads(s)

    assert data["request_id"] == "req-123"
    assert data["method"] == "GET"
    assert data["path"] == "/api/v1/foo"
    assert data["status_code"] == 200
