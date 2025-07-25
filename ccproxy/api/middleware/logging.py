"""Access logging middleware for structured HTTP request/response logging."""

import time
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ccproxy.api.dependencies import get_cached_settings


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

        # Store log storage in request state if collection is enabled

        settings = get_cached_settings(request)

        if settings.observability.logs_collection_enabled and hasattr(
            request.app.state, "log_storage"
        ):
            request.state.log_storage = request.app.state.log_storage

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
        try:
            if hasattr(request.state, "request_id"):
                request_id = request.state.request_id
            elif hasattr(request.state, "context"):
                # Try to check if it's a RequestContext without importing
                context = request.state.context
                if hasattr(context, "request_id") and hasattr(context, "metadata"):
                    request_id = context.request_id
        except Exception:
            # Ignore any errors getting request_id
            pass

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
            try:
                # Calculate duration
                duration_seconds = time.perf_counter() - start_time
                duration_ms = duration_seconds * 1000

                # Extract response info
                if response:
                    status_code = response.status_code

                    # Extract rate limit headers if present
                    rate_limit_info = {}
                    anthropic_request_id = None
                    for header_name, header_value in response.headers.items():
                        header_lower = header_name.lower()
                        # Capture x-ratelimit-* headers
                        if header_lower.startswith(
                            "x-ratelimit-"
                        ) or header_lower.startswith("anthropic-ratelimit-"):
                            rate_limit_info[header_lower] = header_value
                        # Capture request-id from Anthropic's response
                        elif header_lower == "request-id":
                            anthropic_request_id = header_value

                    # Add anthropic request ID if present
                    if anthropic_request_id:
                        rate_limit_info["anthropic_request_id"] = anthropic_request_id

                    # Extract metadata from context if available
                    context_metadata = {}
                    try:
                        if hasattr(request.state, "context"):
                            context = request.state.context
                            # Check if it has the expected attributes of RequestContext
                            if hasattr(context, "metadata") and isinstance(
                                context.metadata, dict
                            ):
                                # Get all metadata from the context
                                context_metadata = context.metadata.copy()
                                # Remove fields we're already logging separately
                                for key in [
                                    "method",
                                    "path",
                                    "client_ip",
                                    "status_code",
                                    "request_id",
                                    "duration_ms",
                                    "duration_seconds",
                                    "query",
                                    "user_agent",
                                    "error_message",
                                ]:
                                    context_metadata.pop(key, None)
                    except Exception:
                        # Ignore any errors extracting context metadata
                        pass

                    # Use start-only logging - let context handle comprehensive access logging
                    # Only log basic request start info since context will handle complete access log
                    from ccproxy.observability.access_logger import log_request_start

                    log_request_start(
                        request_id=request_id or "unknown",
                        method=method,
                        path=path,
                        client_ip=client_ip,
                        user_agent=user_agent,
                        query=query,
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
                        exc_info=True,
                    )
            except Exception as log_error:
                # If logging fails, don't crash the app
                # Use print as a last resort to indicate the issue
                print(f"Failed to write access log: {log_error}")

        return response
