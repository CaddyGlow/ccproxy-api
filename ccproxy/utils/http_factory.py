"""Factory functions for creating pre-configured HTTP clients for different use cases."""

import os
from typing import Any

import httpx

from ccproxy.utils.http_client import (
    HttpClientConfig,
    HttpMiddleware,
    InstrumentedHttpClient,
)
from ccproxy.utils.http_middleware import HttpMetricsMiddleware
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class AuthorizationMiddleware(HttpMiddleware[Any]):
    """Middleware for adding authorization headers to requests."""

    def __init__(self, auth_header: str, auth_value: str):
        """Initialize with authorization header and value.

        Args:
            auth_header: The header name (e.g., 'Authorization', 'X-API-Key')
            auth_value: The header value (e.g., 'Bearer token', 'api-key')
        """
        self.auth_header = auth_header
        self.auth_value = auth_value

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Add authorization header to request."""
        # httpx Request objects are immutable, so we need to rebuild
        # with new headers

        # Create a new headers dict
        headers = dict(request.headers)
        headers[self.auth_header] = self.auth_value

        # Create a new request with updated headers
        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
            params=dict(request.url.params) if request.url.params else None,
        )

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Pass through response unchanged."""
        return response


class RetryMiddleware(HttpMiddleware[Any]):
    """Middleware for enhanced retry logic with exponential backoff."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

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
        """Log retry-able errors."""
        if isinstance(error, ConnectionError | TimeoutError):
            logger.warning(
                f"Retryable error for {request.method} {request.url}: {error}"
            )
        return error


# Removed MetricsMiddleware class - now using HttpMetricsMiddleware from http_middleware


def create_anthropic_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 600.0,
    max_retries: int = 3,
) -> InstrumentedHttpClient | httpx.AsyncClient:
    """Create a pre-configured HTTP client for Anthropic API calls.

    Args:
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        base_url: Base URL for Anthropic API (defaults to https://api.anthropic.com)
        timeout: Request timeout in seconds (default: 600s for long responses)
        max_retries: Maximum number of retries (default: 3)

    Returns:
        InstrumentedHttpClient or httpx.AsyncClient configured for Anthropic API
    """
    # Check feature flag
    try:
        from ccproxy.config import get_settings

        settings = get_settings()
        use_instrumented = getattr(settings, "use_instrumented_http_client", True)
    except Exception:
        # Default to True if settings not available
        use_instrumented = True

    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not api_key:
        logger.warning("No Anthropic API key provided")

    # Set default base URL
    if base_url is None:
        base_url = "https://api.anthropic.com"

    # Get proxy settings from environment
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    # Fall back to regular httpx client if feature flag is disabled
    if not use_instrumented:
        logger.info(
            "Instrumented HTTP client disabled, using regular httpx client for Anthropic API"
        )

        # Create timeout configuration
        timeout_config = httpx.Timeout(
            connect=30.0,
            read=timeout,
            write=timeout,
            pool=timeout,
        )

        # Create limits configuration
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
        )

        # Build headers
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        return httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_config,
            limits=limits,
            proxy=proxy_url,
            headers=headers,
        )

    # Configure middleware for instrumented client
    middleware: list[HttpMiddleware[Any]] = []

    # Add authorization header
    if api_key:
        middleware.append(AuthorizationMiddleware("X-API-Key", api_key))

    # Add retry logic
    middleware.append(RetryMiddleware(max_retries=max_retries, backoff_factor=2.0))

    # Add metrics
    middleware.append(HttpMetricsMiddleware())

    # Create client configuration
    config = HttpClientConfig(
        proxy_url=proxy_url,
        timeout=timeout,
        connect_timeout=30.0,
        max_retries=max_retries,
        retry_backoff=1.0,
        middleware=middleware,
    )

    client = InstrumentedHttpClient(config)
    logger.info(f"Created instrumented Anthropic HTTP client for {base_url}")

    return client


def create_oauth_client(
    client_id: str | None = None,
    client_secret: str | None = None,
    timeout: float = 30.0,
    ssl_verify: bool = True,
) -> InstrumentedHttpClient:
    """Create a pre-configured HTTP client for OAuth authentication flows.

    Args:
        client_id: OAuth client ID
        client_secret: OAuth client secret
        timeout: Request timeout in seconds (default: 30s)
        ssl_verify: Enable SSL verification (default: True)

    Returns:
        InstrumentedHttpClient configured for OAuth flows
    """
    middleware: list[HttpMiddleware[Any]] = []

    # Add basic auth if credentials provided
    if client_id and client_secret:
        import base64

        auth_string = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        middleware.append(
            AuthorizationMiddleware("Authorization", f"Basic {auth_string}")
        )

    # Add retry logic for OAuth endpoints
    middleware.append(RetryMiddleware(max_retries=2, backoff_factor=1.5))

    # Add metrics
    middleware.append(HttpMetricsMiddleware())

    # Get proxy settings
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")

    # Get SSL settings from environment
    ssl_ca_bundle = os.environ.get("SSL_CERT_FILE") or os.environ.get(
        "REQUESTS_CA_BUNDLE"
    )

    # Create client configuration
    config = HttpClientConfig(
        proxy_url=proxy_url,
        ssl_verify=ssl_verify,
        ssl_ca_bundle=ssl_ca_bundle,
        timeout=timeout,
        connect_timeout=10.0,
        max_retries=2,
        retry_backoff=1.5,
        middleware=middleware,
    )

    client = InstrumentedHttpClient(config)
    logger.info("Created OAuth HTTP client")

    return client


def create_internal_client(
    service_name: str = "internal",
    timeout: float = 10.0,
    collect_metrics: bool = False,
) -> InstrumentedHttpClient:
    """Create a pre-configured HTTP client for internal service calls.

    Optimized for low-latency internal calls with minimal overhead.

    Args:
        service_name: Name of the calling service for identification
        timeout: Request timeout in seconds (default: 10s)
        collect_metrics: Enable metrics collection (default: False for performance)

    Returns:
        InstrumentedHttpClient configured for internal calls
    """
    middleware: list[HttpMiddleware[Any]] = []

    # Add metrics if enabled
    if collect_metrics:
        middleware.append(HttpMetricsMiddleware())

    # Minimal retry logic for internal services
    middleware.append(RetryMiddleware(max_retries=1, backoff_factor=1.0))

    # Create client configuration with minimal overhead
    config = HttpClientConfig(
        proxy_url=None,  # No proxy for internal calls
        ssl_verify=False,  # Internal services may use self-signed certs
        timeout=timeout,
        connect_timeout=5.0,
        max_retries=1,
        retry_backoff=1.0,
        middleware=middleware,
        # Higher connection limits for internal services
        max_connections=200,
        max_keepalive_connections=50,
    )

    client = InstrumentedHttpClient(config)
    logger.info(f"Created internal HTTP client for service: {service_name}")

    return client


def create_metrics_client(
    metrics_endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> InstrumentedHttpClient:
    """Create a pre-configured HTTP client for metrics collection.

    Optimized for high-throughput metrics submission with automatic batching.

    Args:
        metrics_endpoint: Metrics collection endpoint URL
        api_key: API key for metrics service
        timeout: Request timeout in seconds (default: 5s for fast metrics)

    Returns:
        InstrumentedHttpClient configured for metrics collection
    """
    from ccproxy.config import get_settings

    settings = get_settings()

    # Get metrics endpoint from settings or parameter
    if metrics_endpoint is None:
        metrics_endpoint = getattr(settings, "metrics_endpoint", None)

    middleware: list[HttpMiddleware[Any]] = []

    # Add authorization if API key provided
    if api_key:
        middleware.append(AuthorizationMiddleware("Authorization", f"Bearer {api_key}"))

    # No retries for metrics to avoid blocking
    middleware.append(RetryMiddleware(max_retries=0, backoff_factor=0))

    # Note: Don't add HttpMetricsMiddleware to avoid circular metrics

    # Create client configuration optimized for metrics
    config = HttpClientConfig(
        proxy_url=None,  # Direct connection for metrics
        ssl_verify=True,
        timeout=timeout,
        connect_timeout=2.0,  # Fast connection timeout
        max_retries=0,  # No retries to avoid blocking
        retry_backoff=0,
        middleware=middleware,
        # High connection limits for metrics throughput
        max_connections=100,
        max_keepalive_connections=20,
    )

    client = InstrumentedHttpClient(config)
    logger.info(f"Created metrics HTTP client for endpoint: {metrics_endpoint}")

    return client


# Convenience factory for general-purpose clients
def create_client(name: str = "default", **kwargs: Any) -> InstrumentedHttpClient:
    """Create a general-purpose HTTP client with custom configuration.

    Args:
        name: Client name for identification
        **kwargs: Additional configuration options passed to HttpClientConfig

    Returns:
        InstrumentedHttpClient with custom configuration
    """
    # Get environment-based proxy settings
    proxy_url = (
        kwargs.get("proxy_url")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("HTTP_PROXY")
    )

    # Get SSL settings
    ssl_verify = kwargs.get("ssl_verify", True)
    ssl_ca_bundle = kwargs.get("ssl_ca_bundle") or os.environ.get("SSL_CERT_FILE")

    # Default middleware
    middleware = kwargs.get("middleware", [])
    if not middleware:
        # Add metrics middleware by default if collect_metrics is True
        collect_metrics = kwargs.get("collect_metrics", True)
        if collect_metrics:
            middleware = [HttpMetricsMiddleware()]
        else:
            middleware = []

    # Create configuration
    config = HttpClientConfig(
        proxy_url=proxy_url,
        ssl_verify=ssl_verify,
        ssl_ca_bundle=ssl_ca_bundle,
        middleware=middleware,
        **{
            k: v
            for k, v in kwargs.items()
            if k not in ["proxy_url", "ssl_verify", "ssl_ca_bundle", "middleware"]
        },
    )

    client = InstrumentedHttpClient(config)
    logger.info(f"Created {name} HTTP client")

    return client
