"""Custom response classes for preserving proxy headers."""

from typing import Any

from fastapi import Response
from starlette.types import Receive, Scope, Send

from ccproxy.core.logging import get_logger


logger = get_logger()


class ProxyResponse(Response):
    """Custom response class that preserves all headers from upstream API.

    This response class ensures that headers like 'server' from the upstream
    API are preserved and not overridden by Uvicorn/Starlette.
    """

    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        media_type: str | None = None,
        background: Any = None,
    ):
        """Initialize the proxy response with preserved headers.

        Args:
            content: Response content
            status_code: HTTP status code
            headers: Headers to preserve from upstream
            media_type: Content type
            background: Background task
        """
        super().__init__(
            content=content,
            status_code=status_code,
            headers=headers,
            media_type=media_type,
            background=background,
        )
        # Store original headers for preservation
        self._preserve_headers = headers or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Override the ASGI call to ensure headers are preserved.

        This method intercepts the response sending process to ensure
        that our headers are not overridden by the server.
        """
        # Build headers preserving original case, avoiding unsafe/duplicate ones
        headers_list: list[tuple[bytes, bytes]] = []
        seen_lower: set[str] = set()

        for name, value in (self._preserve_headers or {}).items():
            lower = name.lower()
            # Skip unsafe or computed headers
            if lower in {"content-length", "transfer-encoding"}:
                continue
            # Avoid duplicates case-insensitively
            if lower in seen_lower:
                continue
            headers_list.append((name.encode(), str(value).encode()))
            seen_lower.add(lower)

        # Add computed Content-Type if missing
        if (
            self.media_type
            and not any(h[0].lower() == b"content-type" for h in headers_list)
        ):
            headers_list.append((b"Content-Type", self.media_type.encode()))

        # Always append correct Content-Length based on actual body
        body_len = len(self.body or b"") if isinstance(self.body, (bytes, bytearray)) else 0
        headers_list.append((b"Content-Length", str(body_len).encode()))

        # Debug logging only; avoid noisy error-level logs for normal responses
        # logger.debug("proxy_response_headers", headers=headers_list)

        await send(
            {
                "type": "http.response.start",
                "status": self.status_code,
                "headers": headers_list,
            }
        )

        await send(
            {
                "type": "http.response.body",
                "body": self.body,
            }
        )
