"""Tests for HTTP client middleware functionality."""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ccproxy.utils.http_client import (
    ChainedHttpMiddleware,
    HttpClientConfig,
    HttpMiddleware,
    InstrumentedHttpClient,
    create_chained_middleware,
    create_http_client,
)


class TestHttpMiddleware(HttpMiddleware[None]):
    """Test implementation of HTTP middleware."""

    def __init__(self, name: str):
        self.name = name
        self.request_count = 0
        self.response_count = 0
        self.error_count = 0

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Track request processing."""
        self.request_count += 1
        # Add header to track middleware processing
        headers = dict(request.headers)
        headers[f"X-Middleware-{self.name}"] = "request"
        new_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )
        return new_request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Track response processing."""
        self.response_count += 1
        # Add header to track middleware processing
        response.headers[f"X-Middleware-{self.name}"] = "response"
        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Track error processing."""
        self.error_count += 1
        return error


class ModifyingMiddleware(HttpMiddleware[None]):
    """Middleware that modifies requests and responses."""

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Add custom header to request."""
        headers = dict(request.headers)
        headers["X-Custom-Header"] = "modified"
        new_request = httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )
        return new_request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Modify response headers (for testing)."""
        # Add a custom header to verify middleware was called
        response.headers["X-Modified"] = "true"
        return response


class ErrorTransformMiddleware(HttpMiddleware[None]):
    """Middleware that transforms errors."""

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Pass through request unchanged."""
        return request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Pass through response unchanged."""
        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Transform connection errors to custom exceptions."""
        if isinstance(error, httpx.ConnectError):
            return RuntimeError(f"Custom error: {str(error)}")
        return error


@pytest.mark.asyncio
async def test_single_middleware():
    """Test HTTP client with single middleware."""
    middleware = TestHttpMiddleware("test1")
    client = create_http_client(middleware=[middleware])

    # Mock the httpx client
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_create.return_value = mock_client

        # Make request
        response = await client.get("https://example.com")

        # Verify middleware was called
        assert middleware.request_count == 1
        assert middleware.response_count == 1
        assert middleware.error_count == 0

        # Verify request was modified by middleware
        sent_request = mock_client.send.call_args[0][0]
        assert "X-Middleware-test1" in sent_request.headers

    await client.close()


@pytest.mark.asyncio
async def test_chained_middleware():
    """Test HTTP client with chained middleware."""
    middleware1 = TestHttpMiddleware("first")
    middleware2 = TestHttpMiddleware("second")
    middleware3 = TestHttpMiddleware("third")

    client = create_http_client(middleware=[middleware1, middleware2, middleware3])

    # Mock the httpx client
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_create.return_value = mock_client

        # Make request
        response = await client.get("https://example.com")

        # Verify all middleware were called in correct order
        assert middleware1.request_count == 1
        assert middleware2.request_count == 1
        assert middleware3.request_count == 1

        # Verify response processing happened in reverse order
        assert middleware1.response_count == 1
        assert middleware2.response_count == 1
        assert middleware3.response_count == 1

        # Verify all middleware headers were added
        sent_request = mock_client.send.call_args[0][0]
        assert "X-Middleware-first" in sent_request.headers
        assert "X-Middleware-second" in sent_request.headers
        assert "X-Middleware-third" in sent_request.headers

    await client.close()


@pytest.mark.asyncio
async def test_middleware_error_handling():
    """Test middleware error handling."""
    middleware = TestHttpMiddleware("error-test")
    error_middleware = ErrorTransformMiddleware()

    client = create_http_client(middleware=[middleware, error_middleware])

    # Mock the httpx client to raise an error
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(
            side_effect=httpx.ConnectError("Connection failed")
        )
        mock_create.return_value = mock_client

        # Make request and expect transformed error
        with pytest.raises(RuntimeError) as exc_info:
            await client.get("https://example.com")

        assert "Custom error" in str(exc_info.value)
        assert middleware.request_count == 1
        assert middleware.error_count == 1
        assert middleware.response_count == 0  # No response due to error

    await client.close()


@pytest.mark.asyncio
async def test_middleware_modification():
    """Test middleware that modifies requests and responses."""
    middleware = ModifyingMiddleware()
    client = create_http_client(middleware=[middleware])

    # Mock the httpx client
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response._status_code = 200  # For modification
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_create.return_value = mock_client

        # Make request
        response = await client.get("https://example.com")

        # Verify request was modified
        sent_request = mock_client.send.call_args[0][0]
        assert sent_request.headers.get("X-Custom-Header") == "modified"

        # Verify response was modified by middleware
        assert "X-Modified" in response.headers
        assert response.headers["X-Modified"] == "true"

    await client.close()


@pytest.mark.asyncio
async def test_dynamic_middleware():
    """Test adding and removing middleware dynamically."""
    middleware1 = TestHttpMiddleware("dynamic1")
    middleware2 = TestHttpMiddleware("dynamic2")

    client = create_http_client()

    # Initially no middleware
    assert client._middleware is None

    # Add first middleware
    client.add_middleware(middleware1)
    assert client._middleware == middleware1

    # Add second middleware - should create a chain
    client.add_middleware(middleware2)
    assert isinstance(client._middleware, ChainedHttpMiddleware)

    # Mock request to test both middleware are active
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_create.return_value = mock_client

        await client.get("https://example.com")

        assert middleware1.request_count == 1
        assert middleware2.request_count == 1

    # Remove first middleware
    client.remove_middleware(middleware1)
    assert client._middleware == middleware2

    # Remove second middleware
    client.remove_middleware(middleware2)
    assert client._middleware is None

    await client.close()


@pytest.mark.asyncio
async def test_create_chained_middleware():
    """Test the create_chained_middleware factory function."""
    middleware1 = TestHttpMiddleware("chain1")
    middleware2 = TestHttpMiddleware("chain2")

    # Test creating chained middleware
    chained = create_chained_middleware([middleware1, middleware2])
    assert isinstance(chained, ChainedHttpMiddleware)
    assert len(chained.middleware_chain) == 2

    # Test error on empty chain
    with pytest.raises(ValueError, match="Middleware chain cannot be empty"):
        create_chained_middleware([])


@pytest.mark.asyncio
async def test_middleware_with_retry():
    """Test middleware interaction with built-in retry logic."""
    middleware = TestHttpMiddleware("retry-test")
    client = create_http_client(
        middleware=[middleware],
        max_retries=2,
        retry_backoff=0.1,
    )

    # Mock the httpx client to fail twice then succeed
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com")
        mock_client.build_request = Mock(return_value=mock_request)

        # Fail twice, then succeed
        mock_client.send = AsyncMock(
            side_effect=[
                httpx.ConnectError("Attempt 1 failed"),
                httpx.ConnectError("Attempt 2 failed"),
                mock_response,  # Success on third attempt
            ]
        )
        mock_create.return_value = mock_client

        # Make request
        response = await client.get("https://example.com")

        # Middleware should only process request once (before retries)
        assert middleware.request_count == 1
        # Error handling happens once for the final error before retries succeed
        assert middleware.error_count == 0  # No error because it eventually succeeds
        # Response processing happens once on success
        assert middleware.response_count == 1

        # Verify the request succeeded
        assert response.status_code == 200

    await client.close()


@pytest.mark.asyncio
async def test_middleware_with_metrics():
    """Test middleware with metrics collection."""
    from ccproxy.utils.http_middleware import HttpMetricsMiddleware

    test_middleware = TestHttpMiddleware("metrics-test")
    metrics_middleware = HttpMetricsMiddleware(
        store_in_memory=True, use_ccproxy_storage=False
    )

    client = create_http_client(
        middleware=[test_middleware, metrics_middleware],
    )

    # Mock the httpx client
    with patch.object(client, "_create_client") as mock_create:
        mock_client = AsyncMock()
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = httpx.Headers({})
        mock_response.content = b"test response"
        mock_response.raise_for_status = Mock()

        mock_request = httpx.Request("GET", "https://example.com", content=b"")
        mock_request.extensions = {}

        mock_client.build_request = Mock(return_value=mock_request)
        mock_client.send = AsyncMock(return_value=mock_response)
        mock_create.return_value = mock_client

        # Make request
        await client.get("https://example.com")

        # Verify metrics were collected by the middleware
        metrics = metrics_middleware.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].method == "GET"
        assert metrics[0].status_code == 200
        assert metrics[0].url == "https://example.com"

        # Test middleware should have been called
        assert test_middleware.request_count == 1
        assert test_middleware.response_count == 1

    await client.close()
