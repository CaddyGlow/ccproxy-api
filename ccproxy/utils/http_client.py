"""HTTP client utilities with integrated metrics and configuration support."""

import asyncio
import contextlib
import json
import ssl
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

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
    is_streaming: bool = Field(
        description="Whether this was a streaming request", default=False
    )
    bytes_streamed: int = Field(description="Total bytes streamed", default=0)

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
        if response:
            # For streaming responses, content is not available
            # Try to get size from Content-Length header if available
            if hasattr(response, "headers") and "content-length" in response.headers:
                try:
                    response_size = int(response.headers["content-length"])
                except (ValueError, TypeError):
                    response_size = 0
            elif hasattr(response, "_content"):
                # For non-streaming responses, content is already loaded
                response_size = len(response._content)
            # For streaming responses without Content-Length, size will be 0
            # The actual size should be set by the caller if tracked during streaming

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

    # Metrics settings removed - use HttpMetricsMiddleware instead

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
        self._middleware: HttpMiddleware[Any] | None = self._setup_middleware()
        self._entered = False

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
            # Metrics are now handled by HttpMetricsMiddleware
            pass

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

    async def stream(
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
        chunk_size: int = 8192,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a streaming HTTP request with metrics collection.

        This method returns an httpx.Response object with stream=True, allowing
        access to headers, status code, and streaming content via aiter_bytes().
        Middleware is applied to the initial request and the final response after
        streaming completes.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            params: Query parameters
            headers: Request headers
            data: Request body data
            json: JSON request body
            files: Files to upload
            timeout: Request timeout in seconds
            chunk_size: Size of chunks to yield (default 8192)
            **kwargs: Additional request parameters

        Returns:
            httpx.Response: Response object with streaming enabled

        Raises:
            Exception: If the request fails or streaming encounters an error
        """
        start_time = time.time()
        request: httpx.Request | None = None
        response: httpx.Response | None = None
        error: str | None = None
        bytes_streamed = 0

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
                        response = await client.send(request, stream=True)
                        break
                    except Exception as e:
                        if attempt == self.config.max_retries:
                            # Apply error middleware before re-raising
                            if self._middleware:
                                e = await self._middleware.process_error(e, request)
                            raise e

                        wait_time = self.config.retry_backoff * (2**attempt)
                        logger.debug(
                            f"Stream request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

                if response is None:
                    raise RuntimeError("No response received")

                # Check for HTTP errors before streaming
                response.raise_for_status()

                # Create a wrapped response that tracks metrics during streaming
                class MetricsTrackingResponse:
                    """Wrapper that tracks metrics during streaming."""

                    def __init__(
                        self, resp: httpx.Response, parent: InstrumentedHttpClient
                    ):
                        self._response = resp
                        self._parent = parent
                        self._bytes_streamed = 0
                        self._closed = False

                    def __getattr__(self, name: str) -> Any:
                        """Delegate attribute access to wrapped response."""
                        return getattr(self._response, name)

                    @property
                    def status_code(self) -> int:
                        """Get response status code."""
                        return self._response.status_code

                    @property
                    def headers(self) -> httpx.Headers:
                        """Get response headers."""
                        return self._response.headers

                    @property
                    def extensions(self) -> dict[str, Any]:
                        """Get response extensions."""
                        return self._response.extensions

                    async def aiter_bytes(
                        self, chunk_size: int | None = None
                    ) -> AsyncIterator[bytes]:
                        """Stream response content as bytes."""
                        if chunk_size is None:
                            chunk_size = 8192

                        error = None
                        try:
                            async for chunk in self._response.aiter_bytes(
                                chunk_size=chunk_size
                            ):
                                self._bytes_streamed += len(chunk)
                                yield chunk
                        except Exception as e:
                            error = e
                            raise
                        finally:
                            # Record metrics when streaming completes (success or failure)
                            if not self._closed:
                                await self._finalize(error=error)

                    async def aclose(self) -> None:
                        """Close the response and record final metrics."""
                        if not self._closed:
                            await self._response.aclose()
                            await self._finalize()
                            self._closed = True

                    async def _finalize(self, error: Exception | None = None) -> None:
                        """Finalize metrics recording."""
                        nonlocal bytes_streamed, response, request, start_time
                        bytes_streamed = self._bytes_streamed

                        # Apply response middleware after streaming completes
                        if self._parent._middleware and not error and request:
                            # Note: middleware can't modify streamed content, but can process metadata
                            await self._parent._middleware.process_response(
                                self._response, request
                            )

                        # Metrics are now handled by HttpMetricsMiddleware
                        # Store streaming info in request extensions for middleware
                        if not hasattr(self, "_metrics_recorded") and request:
                            self._metrics_recorded = True
                            request.extensions["is_streaming"] = True
                            request.extensions["bytes_streamed"] = self._bytes_streamed

                # Return the wrapped response
                return MetricsTrackingResponse(response, self)  # type: ignore[return-value]

        except Exception as e:
            error = str(e)
            logger.error(f"HTTP stream request failed: {method} {url} - {error}")

            # Metrics are now handled by HttpMetricsMiddleware
            pass

            raise

    async def stream_sse(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: Any | None = None,
        json: Any | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Make a Server-Sent Events (SSE) streaming request.

        This method parses SSE format and yields parsed events as dictionaries.

        Args:
            url: Target URL for SSE endpoint
            params: Query parameters
            headers: Request headers (will add Accept: text/event-stream)
            data: Request body data
            json: JSON request body
            timeout: Request timeout in seconds
            **kwargs: Additional request parameters

        Yields:
            dict: Parsed SSE events with 'event', 'data', and optional 'id' fields

        Raises:
            Exception: If the request fails or SSE parsing encounters an error
        """
        # Ensure correct headers for SSE
        if headers is None:
            headers = {}
        headers["Accept"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"

        # Buffer for incomplete SSE messages
        buffer = ""

        response = await self.stream(
            "GET",
            url,
            params=params,
            headers=headers,
            data=data,
            json=json,
            timeout=timeout,
            **kwargs,
        )

        try:
            async for chunk in response.aiter_bytes():
                # Decode chunk and add to buffer
                buffer += chunk.decode("utf-8", errors="ignore")

                # Process complete events in buffer
                while "\n\n" in buffer:
                    event_data, buffer = buffer.split("\n\n", 1)

                    # Parse SSE event
                    event = self._parse_sse_event(event_data)
                    if event:
                        yield event
        finally:
            # Ensure response is closed
            await response.aclose()

    def _parse_sse_event(self, event_data: str) -> dict[str, Any] | None:
        """Parse a single SSE event from raw data.

        Args:
            event_data: Raw SSE event data

        Returns:
            Parsed event dict or None if event is empty/comment
        """
        event: dict[str, Any] = {}

        for line in event_data.strip().split("\n"):
            if not line or line.startswith(":"):
                # Empty line or comment
                continue

            if ":" in line:
                field, value = line.split(":", 1)
                value = value.lstrip()

                if field == "data":
                    # Concatenate multiple data fields
                    if "data" in event:
                        event["data"] += "\n" + value
                    else:
                        event["data"] = value
                elif field in ("event", "id", "retry"):
                    event[field] = value

        # Parse JSON data if possible
        if "data" in event:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                event["parsed_data"] = json.loads(event["data"])

        return event if event else None

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

    async def __aenter__(self) -> "InstrumentedHttpClient":
        """Async context manager entry."""
        self._entered = True
        if self._client is None:
            self._client = self._create_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
        self._entered = False


# Factory function for easy creation
def create_http_client(
    proxy_url: str | None = None,
    ssl_verify: bool = True,
    ssl_ca_bundle: str | None = None,
    timeout: float = 30.0,
    middleware: list[HttpMiddleware[Any]] | None = None,
    **kwargs: Any,
) -> InstrumentedHttpClient | httpx.AsyncClient:
    """Create a configured HTTP client.

    Args:
        proxy_url: HTTP/HTTPS proxy URL
        ssl_verify: Enable SSL verification
        ssl_ca_bundle: Path to CA bundle file
        timeout: Request timeout in seconds
        middleware: List of middleware to apply to requests (use HttpMetricsMiddleware for metrics)
        **kwargs: Additional configuration options

    Returns:
        Configured InstrumentedHttpClient instance or regular httpx.AsyncClient
    """
    # Check feature flag
    try:
        from ccproxy.config import get_settings

        settings = get_settings()
        use_instrumented = getattr(settings, "use_instrumented_http_client", True)
    except Exception:
        # Default to True if settings not available
        use_instrumented = True

    # Fall back to regular httpx client if feature flag is disabled
    if not use_instrumented:
        logger.info("Instrumented HTTP client disabled, using regular httpx client")

        # Create timeout configuration
        timeout_config = httpx.Timeout(
            connect=kwargs.get("connect_timeout", 10.0),
            read=timeout,
            write=timeout,
            pool=timeout,
        )

        # Create limits configuration
        limits = httpx.Limits(
            max_connections=kwargs.get("max_connections", 100),
            max_keepalive_connections=kwargs.get("max_keepalive_connections", 20),
        )

        # Configure SSL
        verify: ssl.SSLContext | str | bool
        if not ssl_verify:
            verify = False
        elif ssl_ca_bundle:
            verify = ssl_ca_bundle
        else:
            verify = True

        # Configure proxy
        proxy = None
        if proxy_url:
            proxy_auth = kwargs.get("proxy_auth")
            if proxy_auth:
                username, password = proxy_auth
                proxy = proxy_url.replace("://", f"://{username}:{password}@")
            else:
                proxy = proxy_url

        return httpx.AsyncClient(
            verify=verify,
            timeout=timeout_config,
            limits=limits,
            proxy=proxy,
        )

    # Use instrumented client
    config = HttpClientConfig(
        proxy_url=proxy_url,
        ssl_verify=ssl_verify,
        ssl_ca_bundle=ssl_ca_bundle,
        timeout=timeout,
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
_global_client: InstrumentedHttpClient | httpx.AsyncClient | None = None


async def get_global_client() -> InstrumentedHttpClient | httpx.AsyncClient:
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
