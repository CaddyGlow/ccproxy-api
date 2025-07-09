"""Metrics middleware for FastAPI request/response interception."""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

from ccproxy.metrics.collector import categorize_user_agent, get_metrics_collector
from ccproxy.metrics.models import ErrorMetrics, HTTPMetrics, UserAgentCategory


logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP request/response metrics."""

    def __init__(self, app: Any) -> None:
        """Initialize the metrics middleware.

        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
        self.metrics_collector = get_metrics_collector()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and collect metrics.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in the chain

        Returns:
            HTTP response with metrics recorded
        """
        # Start timing
        start_time = time.time()

        # Extract basic request info
        method = request.method
        path = request.url.path
        user_agent = request.headers.get("user-agent", "")

        # Determine API type from URL path
        api_type = self._extract_api_type(path)

        # Extract endpoint pattern (remove IDs and query params)
        endpoint = self._extract_endpoint_pattern(path)

        # Categorize user agent
        user_agent_category = categorize_user_agent(user_agent)

        # Calculate request size
        request_size = await self._calculate_request_size(request)

        # Increment active requests
        self.metrics_collector.increment_active_requests(api_type)

        response: Response | None = None
        status_code = 500  # Default to error if something goes wrong
        response_size = 0

        try:
            # Process request
            response = await call_next(request)
            status_code = response.status_code

            # Calculate response size
            response_size = await self._calculate_response_size(response)

        except Exception as exc:
            logger.error(f"Error processing request: {exc}", exc_info=True)

            # Record error metrics
            error_metrics = ErrorMetrics(
                error_type=type(exc).__name__,
                endpoint=endpoint,
                status_code=status_code,
                api_type=api_type,
                user_agent_category=user_agent_category,
            )
            self.metrics_collector.record_error(error_metrics)

            # Store error metrics in database if available
            if (
                hasattr(request.app.state, "metrics_storage")
                and request.app.state.metrics_storage
            ):
                try:
                    # Create HTTP metrics for error case
                    error_http_metrics = HTTPMetrics(
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        api_type=api_type,
                        user_agent_category=user_agent_category,
                        duration_seconds=time.time() - start_time,
                        request_size_bytes=request_size,
                        response_size_bytes=0,
                        user_agent=user_agent,
                        error_type=type(exc).__name__,
                    )
                    request.app.state.metrics_storage.store_request_log(
                        error_http_metrics
                    )
                except Exception as e:
                    logger.warning(f"Failed to store error metrics: {e}")

            # Re-raise the exception
            raise

        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Decrement active requests
            self.metrics_collector.decrement_active_requests(api_type)

            # Record HTTP metrics
            http_metrics = HTTPMetrics(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                api_type=api_type,
                user_agent_category=user_agent_category,
                duration_seconds=duration,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                user_agent=user_agent,
            )
            self.metrics_collector.record_http_request(http_metrics)

            # Store metrics in database if available
            if (
                hasattr(request.app.state, "metrics_storage")
                and request.app.state.metrics_storage
            ):
                try:
                    request.app.state.metrics_storage.store_request_log(http_metrics)
                except Exception as e:
                    logger.warning(f"Failed to store HTTP metrics: {e}")

        # This should never happen in practice since FastAPI middleware guarantees a response
        # The type checker doesn't understand that call_next always returns a response
        assert response is not None
        return response

    def _extract_api_type(self, path: str) -> str:
        """Extract API type from URL path.

        Args:
            path: URL path

        Returns:
            API type string
        """
        if path.startswith("/cc/v1"):
            return "claude_code"
        elif path.startswith("/cc/openai/v1"):
            return "openai_compat"
        elif path.startswith("/api"):
            return "reverse_proxy"
        elif path.startswith("/min"):
            return "minimal_proxy"
        elif path.startswith("/oauth"):
            return "oauth"
        elif path.startswith("/health"):
            return "health"
        else:
            return "reverse_proxy"  # Default fallback

    def _extract_endpoint_pattern(self, path: str) -> str:
        """Extract endpoint pattern from URL path.

        This removes IDs and query parameters to create a consistent pattern.

        Args:
            path: URL path

        Returns:
            Endpoint pattern string
        """
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]

        # Define common patterns to normalize
        patterns = [
            # Replace UUIDs and similar IDs with placeholder
            (
                r"/[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}",
                "/{id}",
            ),
            # Replace numeric IDs
            (r"/\d+", "/{id}"),
            # Replace hash-like strings
            (r"/[a-fA-F0-9]{16,}", "/{id}"),
        ]

        import re

        normalized_path = path
        for pattern, replacement in patterns:
            normalized_path = re.sub(pattern, replacement, normalized_path)

        return normalized_path

    async def _calculate_request_size(self, request: Request) -> int:
        """Calculate request size in bytes.

        Args:
            request: HTTP request

        Returns:
            Request size in bytes
        """
        try:
            # Get the request body
            body = await request.body()
            body_size = len(body) if body else 0

            # Calculate header sizes
            header_size = 0
            for name, value in request.headers.items():
                header_size += (
                    len(name.encode()) + len(value.encode()) + 4
                )  # +4 for ': ' and '\r\n'

            # Add request line size (method + path + version)
            request_line_size = (
                len(request.method.encode()) + len(str(request.url.path).encode()) + 12
            )  # +12 for spaces and HTTP/1.1

            return body_size + header_size + request_line_size

        except Exception as exc:
            logger.warning(f"Failed to calculate request size: {exc}")
            return 0

    async def _calculate_response_size(self, response: Response) -> int:
        """Calculate response size in bytes.

        Args:
            response: HTTP response

        Returns:
            Response size in bytes
        """
        try:
            # Calculate header sizes
            header_size = 0
            for name, value in response.headers.items():
                header_size += (
                    len(name.encode()) + len(value.encode()) + 4
                )  # +4 for ': ' and '\r\n'

            # Add status line size
            status_line_size = (
                len(f"HTTP/1.1 {response.status_code}".encode()) + 4
            )  # +4 for spaces and \r\n

            # For streaming responses, we can't easily calculate body size
            # without consuming the stream, so we'll estimate based on Content-Length
            body_size = 0
            if isinstance(response, StreamingResponse):
                # Try to get content-length header
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        body_size = int(content_length)
                    except ValueError:
                        body_size = 0
            else:
                # For regular responses, we can check if there's a body
                if hasattr(response, "body") and response.body:
                    body_size = len(response.body)

            return header_size + status_line_size + body_size

        except Exception as exc:
            logger.warning(f"Failed to calculate response size: {exc}")
            return 0
