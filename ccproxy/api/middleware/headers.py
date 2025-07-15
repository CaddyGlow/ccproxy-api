"""Header preservation middleware to maintain proxy response headers."""

from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp


class HeaderPreservationMiddleware(BaseHTTPMiddleware):
    """Middleware to preserve certain headers from proxy responses.

    This middleware ensures that headers like 'server' from the upstream
    API are preserved and not overridden by Uvicorn/Starlette.
    """

    def __init__(self, app: ASGIApp):
        """Initialize the header preservation middleware.

        Args:
            app: The ASGI application
        """
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process the request and preserve specific headers.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            The HTTP response with preserved headers
        """
        # Process the request
        response = await call_next(request)

        # Check if we have a stored server header to preserve
        # This would be set by the proxy service if we want to preserve it
        if hasattr(request.state, "preserve_headers"):
            for header_name, header_value in request.state.preserve_headers.items():
                # Force set the header to override any default values
                response.headers[header_name] = header_value
                # Also try raw header setting for more control
                response.raw_headers.append(
                    (header_name.encode(), header_value.encode())
                )

        return response
