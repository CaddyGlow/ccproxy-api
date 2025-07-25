"""Core middleware abstractions for request/response processing."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from ccproxy.core.types import ProxyRequest, ProxyResponse


# Type alias for the next middleware function
NextMiddleware = Callable[[ProxyRequest], Awaitable[ProxyResponse]]


class BaseMiddleware(ABC):
    """Abstract base class for all middleware."""

    @abstractmethod
    async def __call__(
        self, request: ProxyRequest, next: NextMiddleware
    ) -> ProxyResponse:
        """Process the request and call the next middleware.

        Args:
            request: The incoming request
            next: The next middleware in the chain

        Returns:
            The response from the middleware chain

        Raises:
            MiddlewareError: If middleware processing fails
        """
        pass


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol defining the middleware interface."""

    async def __call__(
        self, request: ProxyRequest, next: NextMiddleware
    ) -> ProxyResponse:
        """Process the request and call the next middleware."""
        ...


class MiddlewareChain:
    """Manages a chain of middleware."""

    def __init__(self, middleware: list[BaseMiddleware]):
        """Initialize with a list of middleware.

        Args:
            middleware: List of middleware to apply in order
        """
        self.middleware = middleware

    async def __call__(
        self, request: ProxyRequest, handler: NextMiddleware
    ) -> ProxyResponse:
        """Execute the middleware chain.

        Args:
            request: The incoming request
            handler: The final request handler

        Returns:
            The response from the middleware chain
        """
        # Build the chain from the inside out
        chain = handler
        for mw in reversed(self.middleware):
            # Create a closure to capture the current middleware and chain
            def make_chain(
                current_mw: BaseMiddleware, current_chain: NextMiddleware
            ) -> NextMiddleware:
                async def next_fn(req: ProxyRequest) -> ProxyResponse:
                    return await current_chain(req)

                async def new_chain(req: ProxyRequest) -> ProxyResponse:
                    return await current_mw(req, next_fn)

                return new_chain

            chain = make_chain(mw, chain)

        # Execute the complete chain
        return await chain(request)


class CompositeMiddleware(BaseMiddleware):
    """Middleware that combines multiple middleware into one."""

    def __init__(self, middleware: list[BaseMiddleware]):
        """Initialize with a list of middleware to compose.

        Args:
            middleware: List of middleware to apply in order
        """
        self.chain = MiddlewareChain(middleware)

    async def __call__(
        self, request: ProxyRequest, next: NextMiddleware
    ) -> ProxyResponse:
        """Process the request through all composed middleware.

        Args:
            request: The incoming request
            next: The next middleware in the chain

        Returns:
            The response from the middleware chain
        """
        return await self.chain(request, next)
