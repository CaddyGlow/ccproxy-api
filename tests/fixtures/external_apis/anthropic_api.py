"""External Anthropic API mocks using httpx_mock.

These fixtures intercept HTTP calls to api.anthropic.com for testing ProxyService
and other components that make direct HTTP requests to external APIs.
"""

import json
from collections.abc import Generator
from typing import Any

import pytest
from httpx import HTTPError
from pytest_httpx import HTTPXMock


@pytest.fixture
def mock_external_anthropic_api(
    httpx_mock: HTTPXMock, claude_responses: dict[str, Any]
) -> HTTPXMock:
    """Mock Anthropic API responses for standard completion requests.

    This fixture intercepts HTTP calls to api.anthropic.com and returns
    mock responses for testing ProxyService and similar components.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing HTTP calls to api.anthropic.com
    HTTP Calls: Intercepted and mocked

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception
        claude_responses: Response data fixture

    Returns:
        HTTPXMock configured with Anthropic API responses
    """
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json=claude_responses["standard_completion"],
        status_code=200,
        headers={"content-type": "application/json"},
    )
    return httpx_mock


@pytest.fixture
def mock_external_anthropic_api_streaming(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock Anthropic API streaming responses using SSE format.

    This fixture intercepts HTTP calls to api.anthropic.com for streaming
    responses and returns SSE-formatted mock data.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing streaming HTTP calls to api.anthropic.com
    HTTP Calls: Intercepted and mocked with SSE format

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception

    Returns:
        HTTPXMock configured for SSE streaming responses
    """

    def stream_generator() -> Generator[str, None, None]:
        """Generate SSE formatted streaming response."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        content=b"".join(chunk.encode() for chunk in stream_generator()),
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
        },
    )
    return httpx_mock


@pytest.fixture
def mock_external_anthropic_api_error(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock Anthropic API error responses.

    This fixture intercepts HTTP calls to api.anthropic.com and returns
    error responses for testing error handling in ProxyService.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing error handling for HTTP calls to api.anthropic.com
    HTTP Calls: Intercepted and mocked with error responses

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception

    Returns:
        HTTPXMock configured with error responses
    """
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified",
            },
        },
        status_code=400,
        headers={"content-type": "application/json"},
    )
    return httpx_mock


@pytest.fixture
def mock_external_anthropic_api_unavailable(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock Anthropic API service unavailable responses.

    This fixture intercepts HTTP calls to api.anthropic.com and simulates
    service unavailability for testing resilience.

    Mocking Strategy: External HTTP interception via httpx_mock
    Use Case: Testing service unavailability handling
    HTTP Calls: Intercepted and mocked with 503 responses

    Args:
        httpx_mock: HTTPXMock fixture for HTTP interception

    Returns:
        HTTPXMock configured with service unavailable responses
    """
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json={
            "type": "error",
            "error": {
                "type": "overloaded_error",
                "message": "Service temporarily unavailable",
            },
        },
        status_code=503,
        headers={"content-type": "application/json"},
    )
    return httpx_mock
