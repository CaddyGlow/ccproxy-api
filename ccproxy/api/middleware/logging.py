"""Access logging middleware for structured HTTP request/response logging."""

import time
from typing import Any, Optional

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ccproxy.observability.context import RequestContext


logger = structlog.get_logger(__name__)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Middleware for structured access logging with request/response details."""

    def __init__(self, app: ASGIApp):
        """Initialize the access log middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request and log access details.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The HTTP response
        """
        # Record start time
        start_time = time.perf_counter()

        # Extract client info
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host

        # Extract request info
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else None
        user_agent = request.headers.get("user-agent", "unknown")

        # Get request ID from context if available
        request_id: str | None = None
        if hasattr(request.state, "request_id"):
            request_id = request.state.request_id
        elif hasattr(request.state, "context") and isinstance(
            request.state.context, RequestContext
        ):
            request_id = request.state.context.request_id

        # Process the request
        response: Response | None = None
        error_message: str | None = None

        try:
            response = await call_next(request)
        except Exception as e:
            # Capture error for logging
            error_message = str(e)
            # Re-raise to let error handlers process it
            raise
        finally:
            # Calculate duration
            duration_seconds = time.perf_counter() - start_time
            duration_ms = duration_seconds * 1000

            # Extract response info
            if response:
                status_code = response.status_code

                # Extract rate limit headers if present
                rate_limit_info = {}
                for header_name, header_value in response.headers.items():
                    if header_name.lower().startswith("x-ratelimit-"):
                        rate_limit_info[header_name.lower()] = header_value

                # Log the access
                logger.info(
                    "access_log",
                    request_id=request_id,
                    method=method,
                    path=path,
                    query=query,
                    status_code=status_code,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    duration_ms=duration_ms,
                    duration_seconds=duration_seconds,
                    error_message=error_message,
                    **rate_limit_info,
                )
            else:
                # Log error case
                logger.error(
                    "access_log_error",
                    request_id=request_id,
                    method=method,
                    path=path,
                    query=query,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    duration_ms=duration_ms,
                    duration_seconds=duration_seconds,
                    error_message=error_message or "No response generated",
                )

        return response
