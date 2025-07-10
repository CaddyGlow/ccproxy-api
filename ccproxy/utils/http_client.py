"""HTTP client utilities with integrated metrics and configuration support."""

import asyncio
import ssl
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from ccproxy.config import get_settings
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


# Type variable for middleware return types
T = TypeVar("T")


class HttpMetrics(BaseModel):
    """HTTP request metrics data."""

    url: str = Field(description="Request URL")
    method: str = Field(description="HTTP method")
    status_code: int = Field(description="HTTP status code")
    duration_ms: float = Field(description="Request duration in milliseconds")
    request_size: int = Field(description="Request body size in bytes", default=0)
    response_size: int = Field(description="Response body size in bytes", default=0)
    error: str | None = Field(
        description="Error message if request failed", default=None
    )
    host: str = Field(description="Request host")
    path: str = Field(description="Request path")
    user_agent: str = Field(description="User agent used", default="")

    @classmethod
    def from_request(
        cls,
        request: httpx.Request,
        response: httpx.Response | None = None,
        duration_ms: float = 0,
        error: str | None = None,
    ) -> "HttpMetrics":
        """Create metrics from httpx request and response objects."""
        parsed_url = urlparse(str(request.url))

        # Calculate request size
        request_size = 0
        if request.content:
            request_size = len(request.content)
        elif hasattr(request, "stream") and request.stream:
            # For streaming requests, estimate size
            request_size = 1024  # Default estimate

        # Calculate response size
        response_size = 0
        if response and hasattr(response, "content"):
            response_size = len(response.content)

        return cls(
            url=str(request.url),
            method=request.method,
            status_code=response.status_code if response else 0,
            duration_ms=duration_ms,
            request_size=request_size,
            response_size=response_size,
            error=error,
            host=parsed_url.hostname or "",
            path=parsed_url.path or "/",
            user_agent=request.headers.get("User-Agent", ""),
        )


class HttpClientConfig(BaseModel):
    """HTTP client configuration."""

    model_config = {"arbitrary_types_allowed": True}

    # Proxy settings
    proxy_url: str | None = Field(default=None, description="HTTP/HTTPS proxy URL")
    proxy_auth: tuple[str, str] | None = Field(
        default=None, description="Proxy authentication (username, password)"
    )

    # SSL/TLS settings
    ssl_verify: bool = Field(default=True, description="Enable SSL verification")
    ssl_ca_bundle: str | None = Field(
        default=None, description="Path to CA bundle file"
    )
    ssl_client_cert: str | None = Field(
        default=None, description="Path to client certificate file"
    )
    ssl_client_key: str | None = Field(
        default=None, description="Path to client key file"
    )

    # Timeout settings
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    connect_timeout: float = Field(
        default=10.0, description="Connection timeout in seconds"
    )

    # Connection settings
    max_connections: int = Field(
        default=100, description="Maximum number of connections"
    )
    max_keepalive_connections: int = Field(
        default=20, description="Maximum keepalive connections"
    )

    # Retry settings
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_backoff: float = Field(default=1.0, description="Retry backoff factor")

    # Metrics settings
    collect_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_callback: Any | None = Field(
        default=None, description="Metrics callback function"
    )

    # Middleware settings
    middleware: list["HttpMiddleware[Any]"] = Field(
        default_factory=list,
        description="List of middleware to apply to requests",
        exclude=True,  # Exclude from serialization
    )


class HttpMiddleware(ABC, Generic[T]):
    """Base class for HTTP request/response middleware.

    Middleware can intercept and process HTTP requests before they are sent
    and responses after they are received. This allows for features like
    authentication, retry logic, logging, and more.

    Type parameter T represents the return type of the process methods,
    allowing middleware to transform requests/responses if needed.
    """

    @abstractmethod
    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Process an HTTP request before it is sent.

        Args:
            request: The HTTP request to process

        Returns:
            The processed request (can be modified or a new instance)
        """
        raise NotImplementedError()

    @abstractmethod
    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Process an HTTP response after it is received.

        Args:
            response: The HTTP response to process
            request: The original HTTP request

        Returns:
            The processed response (can be modified or a new instance)
        """
        raise NotImplementedError()

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Process an error that occurred during the request.

        Args:
            error: The exception that occurred
            request: The original HTTP request

        Returns:
            The processed error (can be modified or a new instance)
        """
        # Default implementation just returns the error unchanged
        return error


class ChainedHttpMiddleware(HttpMiddleware[T]):
    """Middleware that chains multiple middleware components together.

    Processes requests and responses through a sequence of middleware components.
    For requests, middleware is applied in order (first to last).
    For responses, middleware is applied in reverse order (last to first).
    For errors, middleware is applied in reverse order (last to first).
    """

    def __init__(self, middleware_chain: list[HttpMiddleware[Any]]) -> None:
        """Initialize chained middleware.

        Args:
            middleware_chain: List of middleware components to chain together.

        Raises:
            ValueError: If middleware_chain is empty
        """
        if not middleware_chain:
            raise ValueError("Middleware chain cannot be empty")

        self.middleware_chain = middleware_chain

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Process request through the middleware chain.

        Args:
            request: The HTTP request to process

        Returns:
            The processed request after passing through all middleware
        """
        current_request = request

        # Process through each middleware in order
        for middleware in self.middleware_chain:
            current_request = await middleware.process_request(current_request)

        return current_request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Process response through the middleware chain.

        Args:
            response: The HTTP response to process
            request: The original HTTP request

        Returns:
            The processed response after passing through all middleware
        """
        current_response = response

        # Process through each middleware in reverse order
        for middleware in reversed(self.middleware_chain):
            current_response = await middleware.process_response(
                current_response, request
            )

        return current_response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Process error through the middleware chain.

        Args:
            error: The exception that occurred
            request: The original HTTP request

        Returns:
            The processed error after passing through all middleware
        """
        current_error = error

        # Process through each middleware in reverse order
        for middleware in reversed(self.middleware_chain):
            current_error = await middleware.process_error(current_error, request)

        return current_error


class InstrumentedHttpClient:
    """HTTP client with integrated metrics and configuration."""

    def __init__(self, config: HttpClientConfig | None = None):
        """Initialize the HTTP client with configuration."""
        self.config = config or HttpClientConfig()
        self._client: httpx.AsyncClient | None = None
        self._metrics: list[HttpMetrics] = []
        self._middleware: HttpMiddleware[Any] | None = self._setup_middleware()

    def _create_ssl_context(self) -> ssl.SSLContext | bool:
        """Create SSL context from configuration."""
        if not self.config.ssl_verify:
            return False  # Disable SSL verification

        context = ssl.create_default_context()

        # Load CA bundle if specified
        if self.config.ssl_ca_bundle:
            ca_path = Path(self.config.ssl_ca_bundle)
            if ca_path.exists():
                context.load_verify_locations(ca_path)
                logger.debug(f"Loaded CA bundle from {ca_path}")
            else:
                logger.warning(f"CA bundle file not found: {ca_path}")

        # Load client certificate if specified
        if self.config.ssl_client_cert:
            cert_path = Path(self.config.ssl_client_cert)
            key_path = (
                Path(self.config.ssl_client_key)
                if self.config.ssl_client_key
                else cert_path
            )

            if cert_path.exists() and key_path.exists():
                context.load_cert_chain(cert_path, key_path)
                logger.debug(f"Loaded client certificate from {cert_path}")
            else:
                logger.warning(
                    f"Client certificate files not found: {cert_path}, {key_path}"
                )

        return context

    def _setup_middleware(self) -> HttpMiddleware[Any] | None:
        """Set up middleware chain from configuration.

        Returns:
            Chained middleware or None if no middleware configured
        """
        if not self.config.middleware:
            return None

        if len(self.config.middleware) == 1:
            return self.config.middleware[0]

        return ChainedHttpMiddleware(self.config.middleware)

    def _create_client(self) -> httpx.AsyncClient:
        """Create configured httpx client."""
        # Create SSL context
        ssl_context = self._create_ssl_context()

        # Create timeout configuration
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.timeout,
            write=self.config.timeout,
            pool=self.config.timeout,
        )

        # Create limits configuration
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
        )

        # Create proxy configuration
        proxy_url = None
        if self.config.proxy_url:
            if self.config.proxy_auth:
                username, password = self.config.proxy_auth
                proxy_url = self.config.proxy_url.replace(
                    "://", f"://{username}:{password}@"
                )
            else:
                proxy_url = self.config.proxy_url
            logger.debug(f"Using proxy: {self.config.proxy_url}")

        return httpx.AsyncClient(
            verify=ssl_context,
            timeout=timeout,
            limits=limits,
            proxy=proxy_url,
        )

    async def _record_metrics(self, metrics: HttpMetrics) -> None:
        """Record metrics for the request."""
        if not self.config.collect_metrics:
            return

        # Store metrics locally
        self._metrics.append(metrics)

        # Keep only last 1000 metrics to prevent memory issues
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]

        # Call custom metrics callback if provided
        if self.config.metrics_callback:
            try:
                await self.config.metrics_callback(metrics)
            except Exception as e:
                logger.error(f"Error in metrics callback: {e}")

        # Integrate with existing metrics system
        try:
            from ccproxy.metrics.database import RequestLog
            from ccproxy.metrics.sync_storage import get_sync_metrics_storage

            settings = get_settings()
            if hasattr(settings, "metrics_enabled") and settings.metrics_enabled:
                storage = get_sync_metrics_storage(
                    f"sqlite:///{settings.metrics_db_path}"
                )

                # Create request log entry
                request_log = RequestLog(
                    method=metrics.method,
                    endpoint=metrics.path,
                    api_type="http_client",
                    status_code=metrics.status_code,
                    duration_ms=metrics.duration_ms,
                    request_size=metrics.request_size,
                    response_size=metrics.response_size,
                    user_agent_category="http_client",
                    error_type=metrics.error,
                    host=metrics.host,
                )

                storage.store_request_log(request_log)
                logger.debug(
                    f"Recorded HTTP client metrics for {metrics.method} {metrics.url}"
                )
        except Exception as e:
            logger.debug(f"Failed to record metrics to storage: {e}")

    @asynccontextmanager
    async def _client_context(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Context manager for HTTP client."""
        if self._client is None:
            self._client = self._create_client()

        try:
            yield self._client
        finally:
            # Don't close the client here, let it be reused
            pass

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: Any | None = None,
        json: Any | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with metrics collection."""
        start_time = time.time()
        request: httpx.Request | None = None
        response: httpx.Response | None = None
        error: str | None = None

        try:
            async with self._client_context() as client:
                # Create request
                request = client.build_request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    data=data,
                    json=json,
                    files=files,
                    timeout=timeout or self.config.timeout,
                    **kwargs,
                )

                # Apply request middleware
                if self._middleware:
                    request = await self._middleware.process_request(request)

                # Make request with retries
                for attempt in range(self.config.max_retries + 1):
                    try:
                        response = await client.send(request)
                        break
                    except Exception as e:
                        if attempt == self.config.max_retries:
                            # Apply error middleware before re-raising
                            if self._middleware:
                                e = await self._middleware.process_error(e, request)
                            raise e

                        wait_time = self.config.retry_backoff * (2**attempt)
                        logger.debug(
                            f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

                # Apply response middleware
                if response and self._middleware:
                    response = await self._middleware.process_response(
                        response, request
                    )

                # Check for HTTP errors
                if response:
                    response.raise_for_status()

        except Exception as e:
            error = str(e)
            logger.error(f"HTTP request failed: {method} {url} - {error}")
            raise

        finally:
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000

            if request:
                metrics = HttpMetrics.from_request(
                    request=request,
                    response=response,
                    duration_ms=duration_ms,
                    error=error,
                )
                await self._record_metrics(metrics)

        if response is None:
            raise RuntimeError("No response received")
        return response

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a PATCH request."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make a HEAD request."""
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs: Any) -> httpx.Response:
        """Make an OPTIONS request."""
        return await self.request("OPTIONS", url, **kwargs)

    def get_metrics(self) -> list[HttpMetrics]:
        """Get collected metrics."""
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self._metrics.clear()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def add_middleware(self, middleware: HttpMiddleware[Any]) -> None:
        """Add middleware to the client.

        Args:
            middleware: Middleware to add to the chain
        """
        self.config.middleware.append(middleware)
        self._middleware = self._setup_middleware()

    def remove_middleware(self, middleware: HttpMiddleware[Any]) -> None:
        """Remove middleware from the client.

        Args:
            middleware: Middleware to remove from the chain
        """
        if middleware in self.config.middleware:
            self.config.middleware.remove(middleware)
            self._middleware = self._setup_middleware()


# Factory function for easy creation
def create_http_client(
    proxy_url: str | None = None,
    ssl_verify: bool = True,
    ssl_ca_bundle: str | None = None,
    timeout: float = 30.0,
    collect_metrics: bool = True,
    middleware: list[HttpMiddleware[Any]] | None = None,
    **kwargs: Any,
) -> InstrumentedHttpClient:
    """Create a configured HTTP client.

    Args:
        proxy_url: HTTP/HTTPS proxy URL
        ssl_verify: Enable SSL verification
        ssl_ca_bundle: Path to CA bundle file
        timeout: Request timeout in seconds
        collect_metrics: Enable metrics collection
        middleware: List of middleware to apply to requests
        **kwargs: Additional configuration options

    Returns:
        Configured InstrumentedHttpClient instance
    """
    config = HttpClientConfig(
        proxy_url=proxy_url,
        ssl_verify=ssl_verify,
        ssl_ca_bundle=ssl_ca_bundle,
        timeout=timeout,
        collect_metrics=collect_metrics,
        middleware=middleware or [],
        **kwargs,
    )
    return InstrumentedHttpClient(config)


def create_chained_middleware(
    middleware_chain: list[HttpMiddleware[Any]],
) -> ChainedHttpMiddleware[Any]:
    """Factory function to create chained middleware.

    Args:
        middleware_chain: List of middleware components to chain together

    Returns:
        ChainedHttpMiddleware instance

    Raises:
        ValueError: If middleware_chain is empty
    """
    return ChainedHttpMiddleware(middleware_chain)


# Global client instance for convenience
_global_client: InstrumentedHttpClient | None = None


async def get_global_client() -> InstrumentedHttpClient:
    """Get or create the global HTTP client."""
    global _global_client
    if _global_client is None:
        _global_client = create_http_client()
    return _global_client


# Convenience functions using the global client
async def get(url: str, **kwargs: Any) -> httpx.Response:
    """Make a GET request using the global client."""
    client = await get_global_client()
    return await client.get(url, **kwargs)


async def post(url: str, **kwargs: Any) -> httpx.Response:
    """Make a POST request using the global client."""
    client = await get_global_client()
    return await client.post(url, **kwargs)


async def put(url: str, **kwargs: Any) -> httpx.Response:
    """Make a PUT request using the global client."""
    client = await get_global_client()
    return await client.put(url, **kwargs)


async def patch(url: str, **kwargs: Any) -> httpx.Response:
    """Make a PATCH request using the global client."""
    client = await get_global_client()
    return await client.patch(url, **kwargs)


async def delete(url: str, **kwargs: Any) -> httpx.Response:
    """Make a DELETE request using the global client."""
    client = await get_global_client()
    return await client.delete(url, **kwargs)
