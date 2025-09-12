"""Format chain injection middleware."""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class FormatChainMiddleware(BaseHTTPMiddleware):
    """Middleware to inject format chain from route decorators into request context."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Inject format chain into request context before calling route handler."""

        # Try to find the route handler and extract format chain
        format_chain = None

        try:
            # Get the route from FastAPI's router
            route = request.scope.get("route")
            if route and hasattr(route, "endpoint"):
                endpoint = route.endpoint

                # Check if the endpoint has a format chain attribute
                if hasattr(endpoint, "__format_chain__"):
                    format_chain = endpoint.__format_chain__

                    logger.debug(
                        "format_chain_injected",
                        path=request.url.path,
                        method=request.method,
                        format_chain=format_chain,
                        category="middleware",
                    )
        except Exception as e:
            logger.debug(
                "format_chain_injection_failed",
                path=request.url.path,
                method=request.method,
                error=str(e),
                category="middleware",
            )

        # Inject format chain into request context if available
        if format_chain:
            # Ensure request.state.context exists
            if not hasattr(request.state, "context"):
                # Create a minimal context object
                import time
                import uuid

                from ccproxy.core.logging import get_logger
                from ccproxy.core.request_context import RequestContext

                request.state.context = RequestContext(
                    request_id=str(uuid.uuid4()),
                    start_time=time.perf_counter(),
                    logger=get_logger(__name__),
                )

            # Set the format chain
            request.state.context.format_chain = format_chain

        # Continue with the request
        response = await call_next(request)
        return response
