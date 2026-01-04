"""Common assertion patterns for API tests.

This module provides reusable assertion functions to reduce duplication
and ensure consistent validation across test files.
"""

from typing import Any

import httpx


def assert_openai_responses_format(data: dict[str, Any]) -> None:
    """Assert that response follows OpenAI API format."""
    required_fields = ["id", "object", "created", "model", "choices", "usage"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Verify types
    assert isinstance(data["id"], str)
    assert isinstance(data["object"], str)
    assert isinstance(data["created"], int)
    assert isinstance(data["model"], str)
    assert isinstance(data["choices"], list)
    assert isinstance(data["usage"], dict)

    # Verify choice structure
    if data["choices"]:
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Verify message structure
        message = choice["message"]
        assert message["role"] == "assistant"
        assert "content" in message


def assert_anthropic_response_format(data: dict[str, Any]) -> None:
    """Assert that response follows Anthropic API format."""
    required_fields = ["id", "type", "role", "content", "model", "stop_reason", "usage"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    # Verify types
    assert isinstance(data["id"], str)
    assert isinstance(data["type"], str)
    assert isinstance(data["role"], str)
    assert isinstance(data["content"], list)
    assert isinstance(data["model"], str)
    assert isinstance(data["usage"], dict)

    # Verify specific values
    assert data["type"] == "message"
    assert data["role"] == "assistant"


def assert_validation_error(response: httpx.Response) -> None:
    """Assert that response is a validation error (422)."""
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert isinstance(data["detail"], list)


def assert_auth_error(response: httpx.Response) -> None:
    """Assert that response is an authentication error (401)."""
    assert response.status_code == 401


def assert_bad_request_error(response: httpx.Response) -> None:
    """Assert that response is a bad request error (400)."""
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def assert_service_unavailable_error(response: httpx.Response) -> None:
    """Assert that response is a service unavailable error (503)."""
    assert response.status_code == 503
    data = response.json()
    assert "error" in data


def assert_sse_headers(response: httpx.Response) -> None:
    """Assert that response has proper SSE headers."""
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"


def assert_sse_format_compliance(chunks: list[str]) -> None:
    """Assert that SSE chunks follow proper format."""
    for chunk in chunks:
        # Skip event: lines, only check data: lines
        if chunk.startswith("event:"):
            continue
        assert chunk.startswith("data: "), (
            f"Chunk should start with 'data: ', got: {chunk}"
        )


def assert_codex_response_format(data: dict[str, Any]) -> None:
    """Assert that response follows updated OpenAI Responses API format."""
    required_fields = [
        "id",
        "object",
        "created_at",
        "model",
        "status",
        "output",
        "parallel_tool_calls",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    assert data["object"] == "response"
    assert isinstance(data["created_at"], int)
    assert isinstance(data["output"], list)
    assert isinstance(data["parallel_tool_calls"], bool)

    if data["output"]:
        item = data["output"][0]
        for field in ("type", "id", "status", "role", "content"):
            assert field in item, f"Missing output field: {field}"
        assert item["type"] == "message"
        assert item["role"] == "assistant"
        assert isinstance(item["content"], list)


def assert_health_response_format(
    data: dict[str, Any], status_values: list[str]
) -> None:
    """Assert that health response follows IETF health format."""
    assert data["status"] in status_values
    assert "version" in data
