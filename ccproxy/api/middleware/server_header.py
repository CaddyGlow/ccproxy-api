"""Server header middleware to set a default server header for non-proxy routes."""

from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ServerHeaderMiddleware:
    """Middleware to set a default server header for responses.

    This middleware adds a server header to responses that don't already have one.
    Proxy responses using ProxyResponse will preserve their upstream server header,
    while other routes will get the default header.
    """

    def __init__(self, app: ASGIApp, server_name: str = "Claude Code Proxy"):
        """Initialize the server header middleware.

        Args:
            app: The ASGI application
            server_name: The default server name to use
        """
        self.app = app
        self.server_name = server_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI application entrypoint."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))

                # Check if server header already exists
                has_server = any(header[0].lower() == b"server" for header in headers)

                # Only add server header for non-proxy routes
                # Proxy routes will have their server header preserved from upstream
                if not has_server:
                    # Check if this looks like a proxy response by looking for specific headers
                    is_proxy_response = any(
                        header[0].lower()
                        in [
                            b"cf-ray",
                            b"cf-cache-status",
                            b"anthropic-ratelimit-unified-status",
                        ]
                        for header in headers
                    )

                    # Only add our server header if this is NOT a proxy response
                    if not is_proxy_response:
                        headers.append((b"server", self.server_name.encode()))
                        message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_wrapper)
