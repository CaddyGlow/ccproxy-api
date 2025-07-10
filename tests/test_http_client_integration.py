"""
Comprehensive integration tests for the migrated HTTP client components.

This module tests the integration of:
- InstrumentedHttpClient with OAuth flows
- Reverse proxy streaming with rate limit handling
- Credentials manager with shared HTTP client
- Middleware chain behavior
- Error scenarios and performance
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

from ccproxy.config.settings import Settings
from ccproxy.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServiceUnavailableError,
)
from ccproxy.metrics.storage import MetricsStorage
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.credentials.models import ClaudeCredentials
from ccproxy.services.credentials.oauth_client import OAuthClient
from ccproxy.services.reverse_proxy import ReverseProxyService
from ccproxy.utils.http_client import (
    ChainedHttpMiddleware,
    HttpClientConfig,
    HttpMetrics,
    HttpMiddleware,
    InstrumentedHttpClient,
    create_chained_middleware,
    create_http_client,
)
from ccproxy.utils.token_extractor import TokenUsage, extract_anthropic_usage


@pytest.fixture
def mock_settings(tmp_path):
    """Create mock settings with temporary paths."""
    return Settings(
        credentials_dir=tmp_path / "credentials",
        metrics_db_path=tmp_path / "metrics.db",
        anthropic_base_url="https://api.anthropic.com",
        anthropic_oauth_client_id="test-client-id",
        anthropic_oauth_client_secret="test-client-secret",
        anthropic_oauth_callback_url="http://localhost:8000/oauth/callback",
    )


@pytest.fixture
def http_client_config():
    """Create HTTP client configuration for testing."""
    return HttpClientConfig(
        timeout=30.0,
        max_retries=3,
        backoff_factor=0.1,
        proxy_url=None,
        verify_ssl=True,
    )


@pytest.fixture
def http_client(http_client_config):
    """Create an instrumented HTTP client."""
    return create_http_client(config=http_client_config)


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for testing."""
    client = Mock(spec=httpx.AsyncClient)
    client.request = AsyncMock()
    client.stream = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
async def metrics_storage(tmp_path):
    """Create a metrics storage instance for testing."""
    storage = MetricsStorage(db_path=tmp_path / "test_metrics.db")
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def valid_credentials():
    """Create valid test credentials."""
    return ClaudeCodeCredentials(
        access_token="valid-access-token",
        refresh_token="valid-refresh-token",
        expires_at=int(time.time()) + 3600,
        profile={
            "email": "test@example.com",
            "name": "Test User",
            "id": "user-123",
        },
    )


@pytest.fixture
def expired_credentials():
    """Create expired test credentials."""
    return ClaudeCodeCredentials(
        access_token="expired-access-token",
        refresh_token="valid-refresh-token",
        expires_at=int(time.time()) - 3600,
        profile={
            "email": "test@example.com",
            "name": "Test User",
            "id": "user-123",
        },
    )


@pytest.mark.integration
class TestOAuthFlowWithInstrumentedClient:
    """Test OAuth flows using the instrumented HTTP client."""

    @pytest.mark.asyncio
    async def test_oauth_token_exchange(self, http_client, mock_settings):
        """Test OAuth token exchange with instrumented client."""
        oauth_client = OAuthClient(settings=mock_settings, http_client=http_client)

        # Mock the HTTP response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.headers = {}

        with patch.object(http_client._client, "request", return_value=mock_response):
            result = await oauth_client.exchange_code_for_token("test-code")

        assert result["access_token"] == "new-access-token"
        assert result["refresh_token"] == "new-refresh-token"
        assert result["expires_in"] == 3600

    @pytest.mark.asyncio
    async def test_oauth_refresh_token(
        self, http_client, mock_settings, expired_credentials
    ):
        """Test OAuth token refresh with instrumented client."""
        oauth_client = OAuthClient(settings=mock_settings, http_client=http_client)

        # Mock the refresh response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed-access-token",
            "refresh_token": "refreshed-refresh-token",
            "expires_in": 7200,
            "token_type": "Bearer",
        }
        mock_response.headers = {}

        with patch.object(http_client._client, "request", return_value=mock_response):
            refreshed = await oauth_client.refresh_token(expired_credentials)

        assert refreshed.access_token == "refreshed-access-token"
        assert refreshed.refresh_token == "refreshed-refresh-token"
        assert refreshed.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_oauth_profile_fetching(
        self, http_client, mock_settings, valid_credentials
    ):
        """Test fetching user profile with instrumented client."""
        oauth_client = OAuthClient(settings=mock_settings, http_client=http_client)

        # Mock the profile response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "user-456",
            "email": "updated@example.com",
            "name": "Updated User",
        }
        mock_response.headers = {}

        with patch.object(http_client._client, "request", return_value=mock_response):
            updated_creds = await oauth_client.fetch_user_profile(valid_credentials)

        assert updated_creds.profile["id"] == "user-456"
        assert updated_creds.profile["email"] == "updated@example.com"
        assert updated_creds.profile["name"] == "Updated User"

    @pytest.mark.asyncio
    async def test_oauth_error_handling(self, http_client, mock_settings):
        """Test OAuth error handling with instrumented client."""
        oauth_client = OAuthClient(settings=mock_settings, http_client=http_client)

        # Mock an error response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "invalid_grant"}
        mock_response.headers = {}

        with patch.object(http_client._client, "request", return_value=mock_response):
            with pytest.raises(AuthenticationError):
                await oauth_client.exchange_code_for_token("invalid-code")


@pytest.mark.integration
class TestReverseProxyStreaming:
    """Test reverse proxy with streaming and rate limit handling."""

    @pytest.fixture
    def proxy_service(self, mock_settings, http_client):
        """Create a reverse proxy service."""
        return ReverseProxyService(settings=mock_settings, http_client=http_client)

    @pytest.mark.asyncio
    async def test_sse_streaming_with_rate_limits(self, proxy_service, http_client):
        """Test SSE streaming with rate limit header preservation."""

        # Mock SSE response with rate limit headers
        async def mock_stream():
            yield b'data: {"type": "message_start", "message": {"id": "msg_123"}}\n\n'
            yield b'data: {"type": "content_block_delta", "delta": {"text": "Hello"}}\n\n'
            yield b'data: {"type": "message_delta", "usage": {"output_tokens": 5}}\n\n'
            yield b'data: {"type": "message_stop"}\n\n'

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {
            "content-type": "text/event-stream",
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "99",
            "anthropic-ratelimit-requests-reset": "2024-01-01T00:00:00Z",
            "anthropic-ratelimit-tokens-limit": "10000",
            "anthropic-ratelimit-tokens-remaining": "9995",
            "anthropic-ratelimit-tokens-reset": "2024-01-01T00:00:00Z",
        }
        mock_response.aiter_bytes = mock_stream
        mock_response.is_streaming = True

        collected_events = []
        rate_limit_headers = {}

        with patch.object(http_client._client, "stream") as mock_stream_method:
            mock_stream_method.return_value.__aenter__.return_value = mock_response

            async for event in proxy_service.proxy_streaming_request(
                path="/v1/messages",
                method="POST",
                headers={"authorization": "Bearer test-key"},
                body={"messages": [{"role": "user", "content": "Hello"}]},
            ):
                if isinstance(event, dict) and "headers" in event:
                    rate_limit_headers.update(event["headers"])
                else:
                    collected_events.append(event)

        # Verify events were collected
        assert len(collected_events) > 0

        # Verify rate limit headers were preserved
        assert "anthropic-ratelimit-requests-limit" in rate_limit_headers
        assert rate_limit_headers["anthropic-ratelimit-requests-limit"] == "100"
        assert "anthropic-ratelimit-tokens-remaining" in rate_limit_headers
        assert rate_limit_headers["anthropic-ratelimit-tokens-remaining"] == "9995"

    @pytest.mark.asyncio
    async def test_token_extraction_from_stream(self, proxy_service, http_client):
        """Test token usage extraction from streaming responses."""

        # Mock streaming response with usage information
        async def mock_stream():
            yield b'data: {"type": "message_start", "message": {"id": "msg_123", "usage": {"input_tokens": 10}}}\n\n'
            yield b'data: {"type": "content_block_delta", "delta": {"text": "Test response"}}\n\n'
            yield b'data: {"type": "message_delta", "usage": {"output_tokens": 3}}\n\n'
            yield b'data: {"type": "message_stop"}\n\n'

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.aiter_bytes = mock_stream
        mock_response.is_streaming = True

        total_usage = TokenUsage()

        with patch.object(http_client._client, "stream") as mock_stream_method:
            mock_stream_method.return_value.__aenter__.return_value = mock_response

            async for event in proxy_service.proxy_streaming_request(
                path="/v1/messages",
                method="POST",
                headers={"authorization": "Bearer test-key"},
                body={"messages": [{"role": "user", "content": "Test"}]},
            ):
                if isinstance(event, bytes):
                    # Parse SSE event
                    event_str = event.decode("utf-8")
                    if event_str.startswith("data: "):
                        try:
                            data = json.loads(event_str[6:])
                            usage = extract_anthropic_usage(data)
                            if usage:
                                total_usage += usage
                        except json.JSONDecodeError:
                            pass

        # Verify token usage was extracted
        assert total_usage.input_tokens == 10
        assert total_usage.output_tokens == 3
        assert total_usage.total_tokens == 13

    @pytest.mark.asyncio
    async def test_streaming_error_propagation(self, proxy_service, http_client):
        """Test error propagation during streaming."""
        # Mock error response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.headers = {
            "anthropic-ratelimit-requests-limit": "100",
            "anthropic-ratelimit-requests-remaining": "0",
            "anthropic-ratelimit-requests-reset": "2024-01-01T00:01:00Z",
            "retry-after": "60",
        }
        mock_response.json.return_value = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded",
            }
        }
        mock_response.is_streaming = False

        with patch.object(http_client._client, "stream") as mock_stream_method:
            mock_stream_method.return_value.__aenter__.return_value = mock_response

            with pytest.raises(RateLimitError) as exc_info:
                async for _ in proxy_service.proxy_streaming_request(
                    path="/v1/messages",
                    method="POST",
                    headers={"authorization": "Bearer test-key"},
                    body={"messages": [{"role": "user", "content": "Test"}]},
                ):
                    pass

            assert "Rate limit exceeded" in str(exc_info.value)


@pytest.mark.integration
class TestCredentialsManagerIntegration:
    """Test credentials manager with shared HTTP client."""

    @pytest.fixture
    async def credentials_manager(self, mock_settings, http_client, tmp_path):
        """Create a credentials manager with shared HTTP client."""
        mock_settings.credentials_dir = tmp_path / "credentials"
        mock_settings.credentials_dir.mkdir(parents=True, exist_ok=True)

        manager = CredentialsManager(settings=mock_settings, http_client=http_client)
        yield manager

    @pytest.mark.asyncio
    async def test_credentials_refresh_with_shared_client(
        self, credentials_manager, expired_credentials, http_client
    ):
        """Test credentials refresh using shared HTTP client."""
        # Save expired credentials
        await credentials_manager.save(expired_credentials)

        # Mock refresh response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }
        mock_response.headers = {}

        with patch.object(http_client._client, "request", return_value=mock_response):
            # Get valid credentials (should trigger refresh)
            valid_creds = await credentials_manager.get_valid_credentials()

        assert valid_creds is not None
        assert valid_creds.access_token == "refreshed-token"
        assert valid_creds.is_valid()

    @pytest.mark.asyncio
    async def test_concurrent_refresh_handling(
        self, credentials_manager, expired_credentials, http_client
    ):
        """Test that concurrent refresh requests only trigger one actual refresh."""
        # Save expired credentials
        await credentials_manager.save(expired_credentials)

        refresh_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal refresh_count
            refresh_count += 1
            await asyncio.sleep(0.1)  # Simulate network delay

            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": f"refreshed-token-{refresh_count}",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
            }
            mock_response.headers = {}
            return mock_response

        with patch.object(http_client._client, "request", side_effect=mock_request):
            # Trigger multiple concurrent refresh attempts
            tasks = [credentials_manager.get_valid_credentials() for _ in range(5)]
            results = await asyncio.gather(*tasks)

        # All results should be the same (from single refresh)
        assert all(r.access_token == results[0].access_token for r in results)
        assert refresh_count == 1  # Only one actual refresh should occur


@pytest.mark.integration
class TestMiddlewareChainBehavior:
    """Test middleware chain behavior with various scenarios."""

    @pytest.mark.asyncio
    async def test_retry_middleware_with_metrics(
        self, http_client_config, metrics_storage
    ):
        """Test retry middleware with metrics collection."""
        # Create client with metrics
        client = create_http_client(
            config=http_client_config, metrics_storage=metrics_storage
        )

        attempt_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            mock_response = Mock(spec=httpx.Response)
            if attempt_count < 3:
                # Fail first two attempts
                mock_response.status_code = 503
                mock_response.headers = {}
                mock_response.json.return_value = {"error": "Service unavailable"}
            else:
                # Succeed on third attempt
                mock_response.status_code = 200
                mock_response.headers = {}
                mock_response.json.return_value = {"result": "success"}

            return mock_response

        with patch.object(client._client, "request", side_effect=mock_request):
            response = await client.request("GET", "https://api.example.com/test")

        assert response.status_code == 200
        assert attempt_count == 3

        # Check metrics were collected
        snapshots = await metrics_storage.get_metrics_snapshots(limit=1)
        assert len(snapshots) > 0
        assert snapshots[0]["http_metrics"]["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_auth_header_middleware(self):
        """Test authentication header middleware."""

        # Create middleware that adds auth headers
        class AuthMiddleware(HttpMiddleware):
            def __init__(self, token: str):
                self.token = token

            async def __call__(self, request, call_next):
                request.headers["Authorization"] = f"Bearer {self.token}"
                return await call_next(request)

        # Create client with auth middleware
        config = HttpClientConfig()
        auth_middleware = AuthMiddleware("test-api-key")
        client = create_http_client(config=config, middleware=[auth_middleware])

        # Mock request to capture headers
        captured_headers = {}

        async def mock_request(method, url, **kwargs):
            captured_headers.update(kwargs.get("headers", {}))
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {"success": True}
            return mock_response

        with patch.object(client._client, "request", side_effect=mock_request):
            await client.request("GET", "https://api.example.com/test")

        assert "Authorization" in captured_headers
        assert captured_headers["Authorization"] == "Bearer test-api-key"

    @pytest.mark.asyncio
    async def test_middleware_error_handling(self):
        """Test middleware error handling and propagation."""

        # Create middleware that raises an error
        class ErrorMiddleware(HttpMiddleware):
            async def __call__(self, request, call_next):
                if request.url.path == "/error":
                    raise ValueError("Middleware error")
                return await call_next(request)

        # Create client with error middleware
        config = HttpClientConfig()
        error_middleware = ErrorMiddleware()
        client = create_http_client(config=config, middleware=[error_middleware])

        # Test error propagation
        with pytest.raises(ValueError, match="Middleware error"):
            await client.request("GET", "https://api.example.com/error")


@pytest.mark.integration
class TestErrorScenarios:
    """Test various error scenarios and recovery."""

    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, http_client):
        """Test recovery from network failures."""
        attempt_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: connection error
                raise httpx.ConnectError("Connection failed")
            elif attempt_count == 2:
                # Second attempt: timeout
                raise httpx.TimeoutException("Request timed out")
            else:
                # Third attempt: success
                mock_response = Mock(spec=httpx.Response)
                mock_response.status_code = 200
                mock_response.headers = {}
                mock_response.json.return_value = {"status": "recovered"}
                return mock_response

        with patch.object(http_client._client, "request", side_effect=mock_request):
            response = await http_client.request("GET", "https://api.example.com/test")

        assert response.status_code == 200
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_timeout_handling(self, http_client_config):
        """Test timeout handling with custom configuration."""
        # Create client with short timeout
        config = HttpClientConfig(timeout=0.1, max_retries=1)
        client = create_http_client(config=config)

        async def slow_request(*args, **kwargs):
            await asyncio.sleep(0.5)  # Longer than timeout
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 200
            return mock_response

        with patch.object(client._client, "request", side_effect=slow_request):
            with pytest.raises(httpx.TimeoutException):
                await client.request("GET", "https://api.example.com/slow")

    @pytest.mark.asyncio
    async def test_http_error_codes(self, http_client):
        """Test handling of various HTTP error codes."""
        error_codes = [400, 401, 403, 404, 429, 500, 502, 503]

        for code in error_codes:
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = code
            mock_response.headers = {}
            mock_response.json.return_value = {"error": f"Error {code}"}

            with patch.object(
                http_client._client, "request", return_value=mock_response
            ):
                response = await http_client.request(
                    "GET", f"https://api.example.com/error/{code}"
                )
                assert response.status_code == code


@pytest.mark.integration
class TestPerformanceComparison:
    """Test performance to ensure no significant overhead."""

    @pytest.mark.asyncio
    async def test_instrumented_client_overhead(self):
        """Compare performance of instrumented vs raw httpx client."""
        # Create instrumented client
        config = HttpClientConfig()
        instrumented_client = create_http_client(config=config)

        # Create raw httpx client
        raw_client = httpx.AsyncClient()

        # Mock fast response
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {"data": "test"}

        # Measure instrumented client
        with patch.object(
            instrumented_client._client, "request", return_value=mock_response
        ):
            start_time = time.time()
            for _ in range(100):
                await instrumented_client.request("GET", "https://api.example.com/test")
            instrumented_time = time.time() - start_time

        # Measure raw client
        with patch.object(raw_client, "request", return_value=mock_response):
            start_time = time.time()
            for _ in range(100):
                await raw_client.request("GET", "https://api.example.com/test")
            raw_time = time.time() - start_time

        await raw_client.aclose()

        # Overhead should be minimal (less than 50%)
        overhead = (instrumented_time - raw_time) / raw_time
        assert overhead < 0.5, f"Excessive overhead: {overhead:.2%}"

    @pytest.mark.asyncio
    async def test_streaming_performance(self, http_client):
        """Test streaming performance with large responses."""
        # Create large streaming response
        chunk_size = 1024
        num_chunks = 1000

        async def mock_stream():
            for _i in range(num_chunks):
                yield b"x" * chunk_size

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/octet-stream"}
        mock_response.aiter_bytes = mock_stream
        mock_response.is_streaming = True

        bytes_received = 0
        start_time = time.time()

        with patch.object(http_client._client, "stream") as mock_stream_method:
            mock_stream_method.return_value.__aenter__.return_value = mock_response

            async with http_client.stream(
                "GET", "https://api.example.com/large"
            ) as response:
                async for chunk in response.aiter_bytes():
                    bytes_received += len(chunk)

        elapsed_time = time.time() - start_time
        throughput_mbps = (bytes_received / elapsed_time) / (1024 * 1024)

        # Should handle at least 100 MB/s (mock throughput)
        assert throughput_mbps > 100
        assert bytes_received == chunk_size * num_chunks


@pytest.mark.unit
class TestMiddlewareUtilities:
    """Test middleware utility functions."""

    def test_create_chained_middleware(self):
        """Test creating chained middleware."""

        # Create test middlewares
        class Middleware1(HttpMiddleware):
            async def __call__(self, request, call_next):
                request.headers["X-Test-1"] = "value1"
                return await call_next(request)

        class Middleware2(HttpMiddleware):
            async def __call__(self, request, call_next):
                request.headers["X-Test-2"] = "value2"
                return await call_next(request)

        middlewares = [Middleware1(), Middleware2()]
        chained = create_chained_middleware(middlewares)

        assert isinstance(chained, ChainedHttpMiddleware)
        assert len(chained.middlewares) == 2

    def test_empty_middleware_chain(self):
        """Test creating empty middleware chain."""
        chained = create_chained_middleware([])
        assert isinstance(chained, ChainedHttpMiddleware)
        assert len(chained.middlewares) == 0


@pytest.mark.unit
class TestHttpClientConfig:
    """Test HTTP client configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HttpClientConfig()
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.backoff_factor == 0.5
        assert config.proxy_url is None
        assert config.verify_ssl is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HttpClientConfig(
            timeout=60.0,
            max_retries=5,
            backoff_factor=1.0,
            proxy_url="http://proxy.example.com:8080",
            verify_ssl=False,
        )
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.backoff_factor == 1.0
        assert config.proxy_url == "http://proxy.example.com:8080"
        assert config.verify_ssl is False
