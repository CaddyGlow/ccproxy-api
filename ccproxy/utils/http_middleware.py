"""Common HTTP middleware implementations for the HTTP client."""

import asyncio
import base64
import time
import uuid
from typing import Any

import httpx

from ccproxy.utils.http_client import HttpMiddleware
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class LoggingMiddleware(HttpMiddleware[Any]):
    """Middleware that logs HTTP requests and responses with appropriate detail.

    This middleware logs:
    - Request method, URL, and headers (with sensitive headers redacted)
    - Request body for POST/PUT/PATCH methods (truncated if too large)
    - Response status code and headers
    - Response body (truncated if too large)
    - Request duration
    - Errors with full context
    """

    def __init__(
        self,
        log_headers: bool = True,
        log_body: bool = True,
        max_body_length: int = 1000,
        redact_headers: list[str] | None = None,
    ):
        """Initialize logging middleware.

        Args:
            log_headers: Whether to log request/response headers
            log_body: Whether to log request/response bodies
            max_body_length: Maximum length of body to log (truncated if longer)
            redact_headers: List of header names to redact (case-insensitive)
        """
        self.log_headers = log_headers
        self.log_body = log_body
        self.max_body_length = max_body_length
        self.redact_headers = [
            h.lower()
            for h in (
                redact_headers
                or [
                    "authorization",
                    "x-api-key",
                    "api-key",
                    "token",
                    "cookie",
                    "set-cookie",
                ]
            )
        ]

    def _redact_headers(self, headers: httpx.Headers) -> dict[str, str]:
        """Redact sensitive headers for logging."""
        result = {}
        for name, value in headers.items():
            if name.lower() in self.redact_headers:
                result[name] = "[REDACTED]"
            else:
                result[name] = value
        return result

    def _truncate_body(self, body: bytes | str | None) -> str:
        """Truncate body for logging if too large."""
        if body is None:
            return "None"

        if isinstance(body, bytes):
            try:
                body = body.decode("utf-8")
            except UnicodeDecodeError:
                return f"<binary data: {len(body)} bytes>"

        if len(body) > self.max_body_length:
            return f"{body[: self.max_body_length]}... (truncated, total length: {len(body)})"
        return body

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Log request details."""
        log_data: dict[str, Any] = {
            "method": request.method,
            "url": str(request.url),
        }

        if self.log_headers:
            log_data["headers"] = self._redact_headers(request.headers)

        if (
            self.log_body
            and request.method in ["POST", "PUT", "PATCH"]
            and request.content
        ):
            log_data["body"] = self._truncate_body(request.content)

        logger.info(f"HTTP Request: {request.method} {request.url}", extra=log_data)

        # Store request timestamp for duration calculation
        request.extensions["request_timestamp"] = time.time()

        return request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Log response details."""
        # Calculate duration
        start_time = request.extensions.get("request_timestamp", time.time())
        duration_ms = (time.time() - start_time) * 1000

        log_data: dict[str, Any] = {
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if self.log_headers:
            log_data["response_headers"] = self._redact_headers(response.headers)

        if self.log_body and response.content:
            log_data["response_body"] = self._truncate_body(response.content)

        logger.info(
            f"HTTP Response: {response.status_code} for {request.method} {request.url} "
            f"({duration_ms:.2f}ms)",
            extra=log_data,
        )

        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Log error details."""
        # Calculate duration
        start_time = request.extensions.get("request_timestamp", time.time())
        duration_ms = (time.time() - start_time) * 1000

        logger.error(
            f"HTTP Error: {type(error).__name__} for {request.method} {request.url} "
            f"({duration_ms:.2f}ms)",
            extra={
                "method": request.method,
                "url": str(request.url),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "duration_ms": round(duration_ms, 2),
            },
            exc_info=True,
        )

        return error


class AuthMiddleware(HttpMiddleware[Any]):
    """Middleware that adds authentication headers to requests.

    Supports multiple authentication schemes:
    - Bearer token authentication
    - API key authentication (header or query parameter)
    - Basic authentication
    - Custom header authentication
    """

    def __init__(
        self,
        bearer_token: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",
        api_key_param: str | None = None,
        basic_auth: tuple[str, str] | None = None,
        custom_headers: dict[str, str] | None = None,
    ):
        """Initialize authentication middleware.

        Args:
            bearer_token: Bearer token for Authorization header
            api_key: API key value
            api_key_header: Header name for API key (default: X-API-Key)
            api_key_param: Query parameter name for API key (if using query auth)
            basic_auth: Tuple of (username, password) for basic auth
            custom_headers: Dictionary of custom headers to add
        """
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.api_key_param = api_key_param
        self.basic_auth = basic_auth
        self.custom_headers = custom_headers or {}

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Add authentication headers to request."""
        headers = dict(request.headers)
        params = dict(request.url.params)

        # Add bearer token
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
            logger.debug("Added bearer token authentication")

        # Add API key to header
        if self.api_key and self.api_key_header and not self.api_key_param:
            headers[self.api_key_header] = self.api_key
            logger.debug(f"Added API key to header: {self.api_key_header}")

        # Add API key to query parameter
        if self.api_key and self.api_key_param:
            params[self.api_key_param] = self.api_key
            logger.debug(f"Added API key to query parameter: {self.api_key_param}")

        # Add basic authentication
        if self.basic_auth:
            username, password = self.basic_auth
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
            logger.debug("Added basic authentication")

        # Add custom headers
        for name, value in self.custom_headers.items():
            headers[name] = value
            logger.debug(f"Added custom header: {name}")

        # Create new request with updated headers and params
        new_url = request.url
        if params != dict(request.url.params):
            new_url = request.url.copy_with(params=params)

        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Check for authentication errors."""
        if response.status_code == 401:
            logger.warning(f"Authentication failed for {request.method} {request.url}")
        elif response.status_code == 403:
            logger.warning(f"Authorization failed for {request.method} {request.url}")

        return response


class RetryMiddleware(HttpMiddleware[Any]):
    """Middleware that implements smart retry logic with exponential backoff.

    Features:
    - Configurable retry conditions (status codes, exceptions)
    - Exponential backoff with jitter
    - Per-request retry tracking
    - Configurable maximum retries and delays
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on_status_codes: set[int] | None = None,
        retry_on_exceptions: tuple[type[Exception], ...] | None = None,
    ):
        """Initialize retry middleware.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add random jitter to delays
            retry_on_status_codes: Set of status codes to retry on
            retry_on_exceptions: Tuple of exception types to retry on
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on_status_codes = retry_on_status_codes or {
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        }
        self.retry_on_exceptions = retry_on_exceptions or (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        delay = min(
            self.initial_delay * (self.exponential_base**attempt), self.max_delay
        )

        if self.jitter:
            # Add up to 25% jitter
            import random

            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay

    def _should_retry_response(self, response: httpx.Response) -> bool:
        """Check if response should be retried."""
        return response.status_code in self.retry_on_status_codes

    def _should_retry_exception(self, error: Exception) -> bool:
        """Check if exception should be retried."""
        return isinstance(error, self.retry_on_exceptions)

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Initialize retry tracking for request."""
        request.extensions["retry_count"] = 0
        return request

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Check if response needs retry."""
        retry_count = request.extensions.get("retry_count", 0)

        if self._should_retry_response(response) and retry_count < self.max_retries:
            delay = self._calculate_delay(retry_count)
            logger.warning(
                f"Retrying request due to status {response.status_code} "
                f"(attempt {retry_count + 1}/{self.max_retries}) "
                f"after {delay:.2f}s delay"
            )

            # Note: Actual retry logic would need to be implemented in the HTTP client
            # This middleware just provides the retry decision and metadata
            response.extensions["should_retry"] = True
            response.extensions["retry_delay"] = delay
            response.extensions["retry_count"] = retry_count + 1

        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Check if error needs retry."""
        retry_count = request.extensions.get("retry_count", 0)

        if self._should_retry_exception(error) and retry_count < self.max_retries:
            delay = self._calculate_delay(retry_count)
            logger.warning(
                f"Retrying request due to {type(error).__name__} "
                f"(attempt {retry_count + 1}/{self.max_retries}) "
                f"after {delay:.2f}s delay",
                exc_info=True,
            )

            # Add retry metadata to the error
            error.extensions = getattr(error, "extensions", {})  # type: ignore
            error.extensions["should_retry"] = True  # type: ignore
            error.extensions["retry_delay"] = delay  # type: ignore
            error.extensions["retry_count"] = retry_count + 1  # type: ignore

        return error


class ProxyAuthMiddleware(HttpMiddleware[Any]):
    """Middleware that handles proxy authentication headers.

    Adds proper proxy authentication headers for HTTP/HTTPS proxies
    that require authentication beyond the URL-embedded credentials.
    """

    def __init__(
        self,
        proxy_username: str,
        proxy_password: str,
        auth_type: str = "basic",
    ):
        """Initialize proxy authentication middleware.

        Args:
            proxy_username: Proxy username
            proxy_password: Proxy password
            auth_type: Authentication type (currently only "basic" is supported)
        """
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.auth_type = auth_type.lower()

        if self.auth_type != "basic":
            raise ValueError(f"Unsupported proxy auth type: {auth_type}")

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Add proxy authentication headers to request."""
        headers = dict(request.headers)

        if self.auth_type == "basic":
            credentials = base64.b64encode(
                f"{self.proxy_username}:{self.proxy_password}".encode()
            ).decode()
            headers["Proxy-Authorization"] = f"Basic {credentials}"
            logger.debug("Added proxy authentication header")

        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Check for proxy authentication errors."""
        if response.status_code == 407:
            logger.error(
                f"Proxy authentication failed for {request.method} {request.url}"
            )

        return response


class RequestIdMiddleware(HttpMiddleware[Any]):
    """Middleware that adds request ID for tracing.

    Adds a unique request ID to each request for correlation and tracing
    across distributed systems. The ID can be used to track requests through
    logs and multiple services.
    """

    def __init__(
        self,
        header_name: str = "X-Request-ID",
        generator: Any | None = None,
        include_in_response: bool = True,
    ):
        """Initialize request ID middleware.

        Args:
            header_name: Header name for the request ID
            generator: Optional custom ID generator function
            include_in_response: Whether to include request ID in response extensions
        """
        self.header_name = header_name
        self.generator = generator or (lambda: str(uuid.uuid4()))
        self.include_in_response = include_in_response

    async def process_request(self, request: httpx.Request) -> httpx.Request:
        """Add request ID header to request."""
        headers = dict(request.headers)

        # Check if request ID already exists
        if self.header_name not in request.headers:
            request_id = self.generator()
            headers[self.header_name] = request_id

            # Store in extensions for later use
            request.extensions["request_id"] = request_id

            logger.debug(f"Added request ID: {request_id}")
        else:
            # Use existing request ID
            request_id = request.headers[self.header_name]
            request.extensions["request_id"] = request_id
            logger.debug(f"Using existing request ID: {request_id}")
            # Return original request since it already has the header
            return request

        return httpx.Request(
            method=request.method,
            url=request.url,
            headers=headers,
            content=request.content,
            extensions=request.extensions,
        )

    async def process_response(
        self, response: httpx.Response, request: httpx.Request
    ) -> httpx.Response:
        """Optionally add request ID to response extensions."""
        if self.include_in_response:
            request_id = request.extensions.get("request_id")
            if request_id:
                response.extensions["request_id"] = request_id
                logger.debug(
                    f"Request {request_id} completed with status {response.status_code}"
                )

        return response

    async def process_error(
        self, error: Exception, request: httpx.Request
    ) -> Exception:
        """Log error with request ID."""
        request_id = request.extensions.get("request_id")
        if request_id:
            logger.error(
                f"Request {request_id} failed with {type(error).__name__}: {error}"
            )

        return error
