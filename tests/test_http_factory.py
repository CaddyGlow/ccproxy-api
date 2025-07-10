"""Tests for HTTP client factory functions."""

import pytest

from ccproxy.utils.http_factory import (
    AuthorizationMiddleware,
    MetricsMiddleware,
    RetryMiddleware,
    create_anthropic_client,
    create_client,
    create_internal_client,
    create_metrics_client,
    create_oauth_client,
)


class TestMiddleware:
    """Test middleware classes."""

    @pytest.mark.asyncio
    async def test_authorization_middleware(self):
        """Test authorization middleware adds headers."""
        from httpx import Request

        middleware = AuthorizationMiddleware("X-API-Key", "test-key")

        # Create a test request
        request = Request("GET", "https://example.com")

        # Process request
        processed = await middleware.process_request(request)

        # Check header was added
        assert processed.headers["X-API-Key"] == "test-key"

    @pytest.mark.asyncio
    async def test_metrics_middleware(self):
        """Test metrics middleware adds service name."""
        from httpx import Request, Response

        middleware = MetricsMiddleware("test-service")

        # Create test request
        request = Request("GET", "https://example.com")

        # Process request
        processed = await middleware.process_request(request)

        # Check service name header
        assert processed.headers["X-Service-Name"] == "test-service"

        # Test response processing (should not error)
        response = Response(200)
        result = await middleware.process_response(response, request)
        assert result == response


class TestFactoryFunctions:
    """Test HTTP client factory functions."""

    def test_create_anthropic_client(self):
        """Test creating Anthropic API client."""
        client = create_anthropic_client(api_key="test-key")

        # Check configuration
        assert client.config.timeout == 600.0
        assert client.config.max_retries == 3
        assert client.config.collect_metrics is True

        # Check middleware was added
        assert len(client.config.middleware) == 3

        # Verify middleware types
        assert isinstance(client.config.middleware[0], AuthorizationMiddleware)
        assert isinstance(client.config.middleware[1], RetryMiddleware)
        assert isinstance(client.config.middleware[2], MetricsMiddleware)

    def test_create_oauth_client(self):
        """Test creating OAuth client."""
        client = create_oauth_client(client_id="test-id", client_secret="test-secret")

        # Check configuration
        assert client.config.timeout == 30.0
        assert client.config.max_retries == 2
        assert client.config.ssl_verify is True

        # Check middleware
        assert len(client.config.middleware) == 3

    def test_create_internal_client(self):
        """Test creating internal service client."""
        client = create_internal_client(service_name="test-service")

        # Check configuration
        assert client.config.timeout == 10.0
        assert client.config.max_retries == 1
        assert client.config.collect_metrics is False
        assert client.config.ssl_verify is False
        assert client.config.proxy_url is None

        # Check connection limits
        assert client.config.max_connections == 200
        assert client.config.max_keepalive_connections == 50

    def test_create_metrics_client(self):
        """Test creating metrics collection client."""
        client = create_metrics_client(api_key="metrics-key")

        # Check configuration
        assert client.config.timeout == 5.0
        assert client.config.connect_timeout == 2.0
        assert client.config.max_retries == 0
        assert client.config.collect_metrics is False
        assert client.config.proxy_url is None

        # Check middleware
        assert len(client.config.middleware) == 3

    def test_create_client_with_defaults(self):
        """Test creating general-purpose client."""
        client = create_client(name="test-client")

        # Check defaults
        assert client.config.ssl_verify is True
        assert len(client.config.middleware) == 1
        assert isinstance(client.config.middleware[0], MetricsMiddleware)

    def test_create_client_with_custom_config(self):
        """Test creating client with custom configuration."""
        client = create_client(
            name="custom", timeout=60.0, max_connections=50, ssl_verify=False
        )

        # Check custom config
        assert client.config.timeout == 60.0
        assert client.config.max_connections == 50
        assert client.config.ssl_verify is False
