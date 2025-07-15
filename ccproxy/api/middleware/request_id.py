"""Request ID middleware for generating and tracking request IDs."""

import time
import uuid
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ccproxy.observability.context import RequestContext


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

        # Create request context
        start_time = time.perf_counter()
        request_logger = logger.bind(request_id=request_id)

        # Create and store request context in state
        request.state.request_id = request_id
        request.state.context = RequestContext(
            request_id=request_id,
            start_time=start_time,
            logger=request_logger,
            metadata={
                "method": request.method,
                "path": str(request.url.path),
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["x-request-id"] = request_id

        return response  # type: ignore[no-any-return]
