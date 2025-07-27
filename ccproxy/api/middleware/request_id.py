"""Request ID middleware for generating and tracking request IDs."""

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ccproxy.observability.context import request_context


logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware for generating request IDs and initializing request context."""

    def __init__(self, app: ASGIApp):
        """Initialize the request ID middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process the request and add request ID/context.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The HTTP response
        """
        # Generate or extract request ID
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

        # Generate datetime for consistent logging across all layers
        log_timestamp = datetime.now(UTC)

        # Get DuckDB storage from app state if available
        storage = getattr(request.app.state, "duckdb_storage", None)

        # Use the proper request context manager to ensure __aexit__ is called
        async with request_context(
            request_id=request_id,
            storage=storage,
            log_timestamp=log_timestamp,
            method=request.method,
            path=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
            query=str(request.url.query) if request.url.query else None,
            service_type="access_log",
        ) as ctx:
            # Store context in request state for access by services
            request.state.request_id = request_id
            request.state.context = ctx

            # Add DuckDB storage to context if available
            if hasattr(request.state, "duckdb_storage"):
                ctx.storage = request.state.duckdb_storage

            # Process the request
            response = await call_next(request)

            # Add request ID to response headers
            response.headers["x-request-id"] = request_id

            return response  # type: ignore[no-any-return]
