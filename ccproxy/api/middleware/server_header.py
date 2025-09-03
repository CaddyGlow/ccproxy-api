"""Server header middleware to set default server and date headers for non-proxy routes."""

from email.utils import formatdate

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ServerHeaderMiddleware:
    """Middleware to set default server and date headers for responses.

    This middleware adds server and date headers to responses that don't already have them.
    Proxy responses will preserve their upstream headers, while other routes will get defaults.
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

                # Check if headers already exist
                has_server = any(header[0].lower() == b"server" for header in headers)
                has_date = any(header[0].lower() == b"date" for header in headers)

                # Add server header if missing
                if not has_server:
                    headers.append((b"server", self.server_name.encode()))

                # Add date header if missing
                if not has_date:
                    date_str = formatdate(timeval=None, localtime=False, usegmt=True)
                    headers.append((b"date", date_str.encode()))

                message["headers"] = headers

            await send(message)

        await self.app(scope, receive, send_wrapper)
