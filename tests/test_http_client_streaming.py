"""Tests for HTTP client streaming functionality."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from ccproxy.utils.http_client import (
    HttpClientConfig,
    HttpMetrics,
    HttpMiddleware,
    InstrumentedHttpClient,
    create_http_client,
)


class StreamingTestMiddleware(HttpMiddleware[None]):
    """Test middleware for streaming requests."""

    def __init__(self):
        self.request_processed = False
        self.response_processed = False
        self.error_processed = False

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Process request and mark as processed."""
        self.request_processed = True
        # Add custom header to verify middleware was applied
        request.headers["X-Test-Middleware"] = "applied"
        return request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Process response and mark as processed."""
        self.response_processed = True
        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Process error and mark as processed."""
        self.error_processed = True
        return error


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient for testing."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def http_client(mock_httpx_client):
    """Create an InstrumentedHttpClient with a mocked httpx client."""
    client = InstrumentedHttpClient()
    client._client = mock_httpx_client
    return client


@pytest.mark.asyncio
async def test_stream_basic(http_client, mock_httpx_client):
    """Test basic streaming functionality."""
    # Mock response that streams chunks
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({})

    # Create async generator for streaming chunks
    async def mock_aiter_bytes(chunk_size):
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        for chunk in chunks:
            yield chunk

    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.raise_for_status = Mock()
    mock_response.aclose = AsyncMock()

    # Mock the request building and sending
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request
    mock_httpx_client.send.return_value = mock_response

    # Get the response object
    response = await http_client.stream("GET", "https://example.com/stream")

    # Verify we got a response object with proper attributes
    assert response.status_code == 200
    assert response.headers == httpx.Headers({})

    # Collect streamed chunks
    chunks = []
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)

    # Verify chunks were received
    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]

    # Verify request was made with streaming enabled
    mock_httpx_client.send.assert_called_once_with(mock_request, stream=True)

    # Close the response
    await response.aclose()

    # Verify metrics were recorded
    metrics = http_client.get_metrics()
    assert len(metrics) == 1
    assert metrics[0].is_streaming is True
    assert metrics[0].bytes_streamed == 18  # Total bytes from all chunks
    assert metrics[0].response_size == 18


@pytest.mark.asyncio
async def test_stream_with_middleware(http_client, mock_httpx_client):
    """Test streaming with middleware applied."""
    # Add test middleware
    middleware = StreamingTestMiddleware()
    http_client.add_middleware(middleware)

    # Mock response
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({})

    async def mock_aiter_bytes(chunk_size):
        yield b"test_chunk"

    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.raise_for_status = Mock()
    mock_response.aclose = AsyncMock()

    # Mock request
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request
    mock_httpx_client.send.return_value = mock_response

    # Get the response object
    response = await http_client.stream("GET", "https://example.com/stream")

    # Collect streamed chunks
    chunks = []
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)

    # Close the response after streaming
    await response.aclose()

    # Verify middleware was applied
    assert middleware.request_processed is True
    assert middleware.response_processed is True
    assert "X-Test-Middleware" in mock_request.headers
    assert mock_request.headers["X-Test-Middleware"] == "applied"


@pytest.mark.asyncio
async def test_stream_error_handling(http_client, mock_httpx_client):
    """Test error handling during streaming."""
    # Mock request
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request

    # Mock connection error
    mock_httpx_client.send.side_effect = httpx.ConnectError("Connection failed")

    # Attempt to stream
    with pytest.raises(httpx.ConnectError):
        await http_client.stream("GET", "https://example.com/stream")

    # Verify metrics recorded the error
    metrics = http_client.get_metrics()
    assert len(metrics) == 1
    assert metrics[0].error == "Connection failed"
    assert metrics[0].is_streaming is True


@pytest.mark.asyncio
async def test_stream_with_retries(http_client, mock_httpx_client):
    """Test streaming with retry logic."""
    # Configure client with retries
    http_client.config.max_retries = 2
    http_client.config.retry_backoff = 0.1

    # Mock response that succeeds on third attempt
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({})

    async def mock_aiter_bytes(chunk_size):
        yield b"success"

    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.raise_for_status = Mock()
    mock_response.aclose = AsyncMock()

    # Mock request
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request

    # Fail twice, then succeed
    mock_httpx_client.send.side_effect = [
        httpx.ConnectError("Attempt 1 failed"),
        httpx.ConnectError("Attempt 2 failed"),
        mock_response,
    ]

    # Stream should succeed after retries
    response = await http_client.stream("GET", "https://example.com/stream")

    chunks = []
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)

    await response.aclose()

    assert chunks == [b"success"]
    assert mock_httpx_client.send.call_count == 3


@pytest.mark.asyncio
async def test_stream_sse():
    """Test Server-Sent Events streaming."""
    # Create a mock response that streams SSE data
    sse_data = [
        b"event: message\n",
        b'data: {"type": "greeting", "text": "Hello"}\n\n',
        b"event: update\n",
        b'data: {"progress": 50}\n',
        b"id: 123\n\n",
        b"data: plain text message\n\n",
        b": comment line\n\n",
        b"event: close\n",
        b'data: {"status": "complete"}\n\n',
    ]

    # Create client with mocked stream method
    client = InstrumentedHttpClient()

    # Create a mock response that returns SSE data
    class MockSSEResponse:
        def __init__(self, data):
            self.data = data
            self.status_code = 200
            self.headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            for chunk in self.data:
                yield chunk

        async def aclose(self):
            pass

    async def mock_stream(*args, **kwargs):
        return MockSSEResponse(sse_data)

    # Replace stream method with mock
    client.stream = mock_stream  # type: ignore[method-assign]

    # Collect SSE events
    events = []
    async for event in client.stream_sse("https://example.com/sse"):
        events.append(event)

    # Verify parsed events
    assert len(events) == 4

    # First event
    assert events[0]["event"] == "message"
    assert events[0]["data"] == '{"type": "greeting", "text": "Hello"}'
    assert events[0]["parsed_data"] == {"type": "greeting", "text": "Hello"}

    # Second event
    assert events[1]["event"] == "update"
    assert events[1]["data"] == '{"progress": 50}'
    assert events[1]["parsed_data"] == {"progress": 50}
    assert events[1]["id"] == "123"

    # Third event (plain text, no JSON parsing)
    assert events[2]["data"] == "plain text message"
    assert "parsed_data" not in events[2]

    # Fourth event
    assert events[3]["event"] == "close"
    assert events[3]["parsed_data"] == {"status": "complete"}


@pytest.mark.asyncio
async def test_stream_sse_with_multiline_data():
    """Test SSE streaming with multiline data fields."""
    # SSE data with multiline data field
    sse_data = [
        b"event: multiline\n",
        b"data: line1\n",
        b"data: line2\n",
        b"data: line3\n\n",
    ]

    client = InstrumentedHttpClient()

    # Create a mock response that returns SSE data
    class MockSSEResponse:
        def __init__(self, data):
            self.data = data
            self.status_code = 200
            self.headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            for chunk in self.data:
                yield chunk

        async def aclose(self):
            pass

    async def mock_stream(*args, **kwargs):
        return MockSSEResponse(sse_data)

    client.stream = mock_stream  # type: ignore[method-assign]

    events = []
    async for event in client.stream_sse("https://example.com/sse"):
        events.append(event)

    assert len(events) == 1
    assert events[0]["event"] == "multiline"
    assert events[0]["data"] == "line1\nline2\nline3"


@pytest.mark.asyncio
async def test_stream_sse_headers():
    """Test that SSE requests have correct headers."""
    client = InstrumentedHttpClient()

    # Track headers passed to stream method
    captured_headers = None

    # Create a mock response
    class MockSSEResponse:
        def __init__(self):
            self.status_code = 200
            self.headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            yield b"data: test\n\n"

        async def aclose(self):
            pass

    async def mock_stream(*args, **kwargs):
        nonlocal captured_headers
        captured_headers = kwargs.get("headers", {})
        return MockSSEResponse()

    client.stream = mock_stream  # type: ignore[method-assign]

    # Make SSE request
    async for _event in client.stream_sse("https://example.com/sse"):
        pass

    # Verify SSE headers were set
    assert captured_headers is not None
    assert captured_headers["Accept"] == "text/event-stream"
    assert captured_headers["Cache-Control"] == "no-cache"


@pytest.mark.asyncio
async def test_stream_http_error():
    """Test streaming with HTTP error response."""
    client = InstrumentedHttpClient()

    # Mock response with error status
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 404
    mock_response.headers = httpx.Headers({})
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=Mock(), response=mock_response
    )

    # Mock client context and request
    mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request
    mock_httpx_client.send.return_value = mock_response

    client._client = mock_httpx_client

    # Attempt to stream
    with pytest.raises(httpx.HTTPStatusError):
        await client.stream("GET", "https://example.com/stream")

    # Verify metrics recorded the error
    metrics = client.get_metrics()
    assert len(metrics) == 1
    assert metrics[0].error is not None and "Not Found" in metrics[0].error
    assert metrics[0].status_code == 404


@pytest.mark.asyncio
async def test_stream_metrics_collection():
    """Test that streaming collects accurate metrics."""
    client = InstrumentedHttpClient()

    # Mock response with specific chunk sizes
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({"Content-Type": "text/plain"})

    chunk_sizes = [1024, 2048, 512]  # Total: 3584 bytes

    async def mock_aiter_bytes(chunk_size):
        for size in chunk_sizes:
            yield b"x" * size

    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.raise_for_status = Mock()
    mock_response.aclose = AsyncMock()

    # Mock client
    mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/large-stream"
    mock_request.method = "POST"
    mock_request.headers = httpx.Headers({"User-Agent": "test-agent"})
    mock_request.content = b"request_body"

    mock_httpx_client.build_request.return_value = mock_request
    mock_httpx_client.send.return_value = mock_response

    client._client = mock_httpx_client

    # Stream and collect all data
    response = await client.stream("POST", "https://example.com/large-stream")

    total_bytes = 0
    async for chunk in response.aiter_bytes():
        total_bytes += len(chunk)

    await response.aclose()

    # Verify total bytes
    assert total_bytes == 3584

    # Check metrics
    metrics = client.get_metrics()
    assert len(metrics) == 1

    metric = metrics[0]
    assert metric.url == "https://example.com/large-stream"
    assert metric.method == "POST"
    assert metric.status_code == 200
    assert metric.is_streaming is True
    assert metric.bytes_streamed == 3584
    assert metric.response_size == 3584
    assert metric.request_size == 12  # len(b"request_body")
    assert metric.error is None
    assert metric.user_agent == "test-agent"
    assert metric.duration_ms > 0


@pytest.mark.asyncio
async def test_stream_partial_failure():
    """Test streaming that fails partway through."""
    client = InstrumentedHttpClient()

    # Mock response that fails during streaming
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = httpx.Headers({})

    async def mock_aiter_bytes(chunk_size):
        yield b"chunk1"
        yield b"chunk2"
        raise httpx.NetworkError("Connection lost")

    mock_response.aiter_bytes = mock_aiter_bytes
    mock_response.raise_for_status = Mock()
    mock_response.aclose = AsyncMock()

    # Mock client
    mock_httpx_client = AsyncMock(spec=httpx.AsyncClient)
    mock_request = Mock(spec=httpx.Request)
    mock_request.url = "https://example.com/stream"
    mock_request.method = "GET"
    mock_request.headers = httpx.Headers({"User-Agent": "test"})
    mock_request.content = b""

    mock_httpx_client.build_request.return_value = mock_request
    mock_httpx_client.send.return_value = mock_response

    client._client = mock_httpx_client

    # Attempt to stream
    chunks_received = []
    with pytest.raises(httpx.NetworkError):
        response = await client.stream("GET", "https://example.com/stream")
        async for chunk in response.aiter_bytes():
            chunks_received.append(chunk)

    # Verify we got some chunks before failure
    assert chunks_received == [b"chunk1", b"chunk2"]

    # Verify metrics show partial streaming
    metrics = client.get_metrics()
    assert len(metrics) == 1
    assert metrics[0].bytes_streamed == 12  # Two chunks received
    assert metrics[0].error is not None and "Connection lost" in metrics[0].error
