"""Tests for HTTP middleware implementations."""

import asyncio
import base64
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

from ccproxy.utils.http_middleware import (
    AuthMiddleware,
    LoggingMiddleware,
    ProxyAuthMiddleware,
    RequestIdMiddleware,
    RetryMiddleware,
)


@pytest.mark.asyncio
async def test_logging_middleware_request():
    """Test LoggingMiddleware request processing."""
    middleware = LoggingMiddleware(
        log_headers=True,
        log_body=True,
        max_body_length=50,
        redact_headers=["authorization", "x-api-key"],
    )

    # Create test request
    request = httpx.Request(
        method="POST",
        url="https://example.com/api/test",
        headers={
            "Authorization": "Bearer secret-token",
            "X-Api-Key": "secret-key",
            "Content-Type": "application/json",
        },
        content=b'{"message": "This is a test message that is longer than 50 characters"}',
    )

    # Process request
    processed_request = await middleware.process_request(request)

    # Verify request is returned unchanged
    assert processed_request.method == request.method
    assert processed_request.url == request.url

    # Verify timestamp was added
    assert "request_timestamp" in processed_request.extensions


@pytest.mark.asyncio
async def test_logging_middleware_response():
    """Test LoggingMiddleware response processing."""
    middleware = LoggingMiddleware()

    # Create test request with timestamp
    request = httpx.Request("GET", "https://example.com")
    request.extensions["request_timestamp"] = 1234567890.0

    # Create test response
    response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=b'{"result": "success"}',
    )

    # Mock time to control duration calculation
    with patch("time.time", return_value=1234567891.0):
        processed_response = await middleware.process_response(response, request)

    # Verify response is returned unchanged
    assert processed_response.status_code == response.status_code


@pytest.mark.asyncio
async def test_auth_middleware_bearer_token():
    """Test AuthMiddleware with bearer token."""
    middleware = AuthMiddleware(bearer_token="test-token")

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    assert processed_request.headers["Authorization"] == "Bearer test-token"


@pytest.mark.asyncio
async def test_auth_middleware_api_key_header():
    """Test AuthMiddleware with API key in header."""
    middleware = AuthMiddleware(
        api_key="test-api-key",
        api_key_header="X-Custom-API-Key",
    )

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    assert processed_request.headers["X-Custom-API-Key"] == "test-api-key"


@pytest.mark.asyncio
async def test_auth_middleware_api_key_param():
    """Test AuthMiddleware with API key in query parameter."""
    middleware = AuthMiddleware(
        api_key="test-api-key",
        api_key_param="api_key",
    )

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    assert processed_request.url.params["api_key"] == "test-api-key"


@pytest.mark.asyncio
async def test_auth_middleware_basic_auth():
    """Test AuthMiddleware with basic authentication."""
    middleware = AuthMiddleware(basic_auth=("user", "pass"))

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    # Verify basic auth header
    expected_creds = base64.b64encode(b"user:pass").decode()
    assert processed_request.headers["Authorization"] == f"Basic {expected_creds}"


@pytest.mark.asyncio
async def test_auth_middleware_custom_headers():
    """Test AuthMiddleware with custom headers."""
    middleware = AuthMiddleware(
        custom_headers={
            "X-Custom-Header": "custom-value",
            "X-Another-Header": "another-value",
        }
    )

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    assert processed_request.headers["X-Custom-Header"] == "custom-value"
    assert processed_request.headers["X-Another-Header"] == "another-value"


@pytest.mark.asyncio
async def test_retry_middleware_response():
    """Test RetryMiddleware response processing."""
    middleware = RetryMiddleware(
        max_retries=3,
        retry_on_status_codes={500, 503},
    )

    request = httpx.Request("GET", "https://example.com")
    request.extensions["retry_count"] = 0

    # Test retryable status code
    response = httpx.Response(status_code=503)
    processed_response = await middleware.process_response(response, request)

    assert processed_response.extensions["should_retry"] is True
    assert processed_response.extensions["retry_delay"] > 0
    assert processed_response.extensions["retry_count"] == 1

    # Test non-retryable status code
    response = httpx.Response(status_code=404)
    processed_response = await middleware.process_response(response, request)

    assert "should_retry" not in processed_response.extensions


@pytest.mark.asyncio
async def test_retry_middleware_error():
    """Test RetryMiddleware error processing."""
    middleware = RetryMiddleware(
        max_retries=2,
        retry_on_exceptions=(httpx.ConnectError, httpx.TimeoutException),
    )

    request = httpx.Request("GET", "https://example.com")
    request.extensions["retry_count"] = 0

    # Test retryable exception
    error = httpx.ConnectError("Connection failed")
    processed_error = await middleware.process_error(error, request)

    assert hasattr(processed_error, "extensions")
    assert processed_error.extensions["should_retry"] is True  # type: ignore
    assert processed_error.extensions["retry_delay"] > 0  # type: ignore
    assert processed_error.extensions["retry_count"] == 1  # type: ignore

    # Test non-retryable exception
    error2 = ValueError("Invalid value")
    processed_error = await middleware.process_error(error2, request)

    # Should not have extensions added
    assert not hasattr(processed_error, "extensions") or "should_retry" not in getattr(
        processed_error, "extensions", {}
    )


@pytest.mark.asyncio
async def test_retry_middleware_max_retries():
    """Test RetryMiddleware respects max retries."""
    middleware = RetryMiddleware(max_retries=2)

    request = httpx.Request("GET", "https://example.com")
    request.extensions["retry_count"] = 2  # Already at max

    response = httpx.Response(status_code=503)
    processed_response = await middleware.process_response(response, request)

    # Should not retry when at max retries
    assert "should_retry" not in processed_response.extensions


@pytest.mark.asyncio
async def test_proxy_auth_middleware():
    """Test ProxyAuthMiddleware."""
    middleware = ProxyAuthMiddleware(
        proxy_username="proxy-user",
        proxy_password="proxy-pass",
    )

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    # Verify proxy auth header
    expected_creds = base64.b64encode(b"proxy-user:proxy-pass").decode()
    assert processed_request.headers["Proxy-Authorization"] == f"Basic {expected_creds}"


@pytest.mark.asyncio
async def test_proxy_auth_middleware_invalid_type():
    """Test ProxyAuthMiddleware with invalid auth type."""
    with pytest.raises(ValueError, match="Unsupported proxy auth type"):
        ProxyAuthMiddleware(
            proxy_username="user",
            proxy_password="pass",
            auth_type="digest",
        )


@pytest.mark.asyncio
async def test_request_id_middleware_new_id():
    """Test RequestIdMiddleware generates new ID."""
    test_uuid = "test-uuid-1234"
    middleware = RequestIdMiddleware(
        header_name="X-Request-ID",
        generator=lambda: test_uuid,
    )

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    assert processed_request.headers["X-Request-ID"] == test_uuid
    assert processed_request.extensions["request_id"] == test_uuid


@pytest.mark.asyncio
async def test_request_id_middleware_existing_id():
    """Test RequestIdMiddleware preserves existing ID."""
    middleware = RequestIdMiddleware()

    existing_id = "existing-request-id"
    request = httpx.Request(
        "GET",
        "https://example.com",
        headers={"X-Request-ID": existing_id},
    )
    processed_request = await middleware.process_request(request)

    assert processed_request.headers["X-Request-ID"] == existing_id
    assert processed_request.extensions["request_id"] == existing_id


@pytest.mark.asyncio
async def test_request_id_middleware_response():
    """Test RequestIdMiddleware adds ID to response."""
    middleware = RequestIdMiddleware(include_in_response=True)

    request = httpx.Request("GET", "https://example.com")
    request.extensions["request_id"] = "test-id"

    response = httpx.Response(status_code=200)
    processed_response = await middleware.process_response(response, request)

    assert processed_response.extensions["request_id"] == "test-id"


@pytest.mark.asyncio
async def test_request_id_middleware_response_no_include():
    """Test RequestIdMiddleware without including ID in response."""
    middleware = RequestIdMiddleware(include_in_response=False)

    request = httpx.Request("GET", "https://example.com")
    request.extensions["request_id"] = "test-id"

    response = httpx.Response(status_code=200)
    processed_response = await middleware.process_response(response, request)

    assert "request_id" not in processed_response.extensions


@pytest.mark.asyncio
async def test_request_id_middleware_default_generator():
    """Test RequestIdMiddleware with default UUID generator."""
    middleware = RequestIdMiddleware()

    request = httpx.Request("GET", "https://example.com")
    processed_request = await middleware.process_request(request)

    # Verify a UUID was generated
    request_id = processed_request.headers["X-Request-ID"]
    assert request_id
    # Simple check that it looks like a UUID
    assert len(request_id) == 36
    assert request_id.count("-") == 4
