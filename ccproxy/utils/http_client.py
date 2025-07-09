"""HTTP client utilities with integrated metrics and configuration support."""

import asyncio
import ssl
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from ccproxy.config import get_settings
from ccproxy.utils.logging import get_logger

logger = get_logger(__name__)


class HttpMetrics(BaseModel):
    """HTTP request metrics data."""

    url: str = Field(description="Request URL")
    method: str = Field(description="HTTP method")
    status_code: int = Field(description="HTTP status code")
    duration_ms: float = Field(description="Request duration in milliseconds")
    request_size: int = Field(description="Request body size in bytes", default=0)
    response_size: int = Field(description="Response body size in bytes", default=0)
    error: Optional[str] = Field(
        description="Error message if request failed", default=None
    )
    host: str = Field(description="Request host")
    path: str = Field(description="Request path")
    user_agent: str = Field(description="User agent used", default="")

    @classmethod
    def from_request(
        cls,
        request: httpx.Request,
        response: Optional[httpx.Response] = None,
        duration_ms: float = 0,
        error: Optional[str] = None,
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

    # Proxy settings
    proxy_url: Optional[str] = Field(default=None, description="HTTP/HTTPS proxy URL")
    proxy_auth: Optional[tuple[str, str]] = Field(
        default=None, description="Proxy authentication (username, password)"
    )

    # SSL/TLS settings
    ssl_verify: bool = Field(default=True, description="Enable SSL verification")
    ssl_ca_bundle: Optional[str] = Field(
        default=None, description="Path to CA bundle file"
    )
    ssl_client_cert: Optional[str] = Field(
        default=None, description="Path to client certificate file"
    )
    ssl_client_key: Optional[str] = Field(
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
    metrics_callback: Optional[Any] = Field(
        default=None, description="Metrics callback function"
    )


class InstrumentedHttpClient:
    """HTTP client with integrated metrics and configuration."""

    def __init__(self, config: Optional[HttpClientConfig] = None):
        """Initialize the HTTP client with configuration."""
        self.config = config or HttpClientConfig()
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics: list[HttpMetrics] = []

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
            from ccproxy.metrics.sync_storage import get_sync_metrics_storage
            from ccproxy.metrics.database import RequestLog

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
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        files: Optional[dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with metrics collection."""
        start_time = time.time()
        request: Optional[httpx.Request] = None
        response: Optional[httpx.Response] = None
        error: Optional[str] = None

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

                # Make request with retries
                for attempt in range(self.config.max_retries + 1):
                    try:
                        response = await client.send(request)
                        break
                    except Exception as e:
                        if attempt == self.config.max_retries:
                            raise

                        wait_time = self.config.retry_backoff * (2**attempt)
                        logger.debug(
                            f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}"
                        )
                        await asyncio.sleep(wait_time)

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


# Factory function for easy creation
def create_http_client(
    proxy_url: Optional[str] = None,
    ssl_verify: bool = True,
    ssl_ca_bundle: Optional[str] = None,
    timeout: float = 30.0,
    collect_metrics: bool = True,
    **kwargs: Any,
) -> InstrumentedHttpClient:
    """Create a configured HTTP client."""
    config = HttpClientConfig(
        proxy_url=proxy_url,
        ssl_verify=ssl_verify,
        ssl_ca_bundle=ssl_ca_bundle,
        timeout=timeout,
        collect_metrics=collect_metrics,
        **kwargs,
    )
    return InstrumentedHttpClient(config)


# Global client instance for convenience
_global_client: Optional[InstrumentedHttpClient] = None


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
