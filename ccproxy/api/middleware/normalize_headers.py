from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from ccproxy.core.logging import get_logger


logger = get_logger()


class NormalizeHeadersMiddleware:
    """Middleware to normalize outgoing response headers.

    - Strips unsafe/mismatched headers (Content-Length, Transfer-Encoding)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        send_called = False

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal send_called
            if message.get("type") == "http.response.start":
                headers = message.get("headers", [])
                # Filter out content-length and transfer-encoding
                filtered: list[tuple[bytes, bytes]] = []
                has_server = False
                for name, value in headers:
                    lower = name.lower()
                    if lower in (b"content-length", b"transfer-encoding"):
                        continue
                    if lower == b"server":
                        has_server = True
                    filtered.append((name, value))

                # Ensure a Server header exists; default to "ccproxy"
                if not has_server:
                    filtered.append((b"server", b"ccproxy"))

                message = {**message, "headers": filtered}
                send_called = True
            await send(message)

        # Call downstream app
        await self.app(scope, receive, send_wrapper)

        # Note: We are not re-wrapping to ProxyResponse here because we operate
        # at ASGI message level. Header normalization is sufficient; Starlette
        # computes Content-Length automatically from body when omitted.
        return
