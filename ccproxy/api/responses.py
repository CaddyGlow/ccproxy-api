"""Custom response classes for preserving proxy headers."""

from typing import Any

from fastapi import Response
from starlette.types import Receive, Scope, Send


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
        # Ensure we include all original headers, including 'server'
        headers_list = []

        # Add all headers from the response, but skip content-length
        # as we'll recalculate it based on actual body
        for name, value in self._preserve_headers.items():
            lower_name = name.lower()
            # Skip content-length and transfer-encoding as we'll set them correctly
            if lower_name not in ["content-length", "transfer-encoding"]:
                headers_list.append((lower_name.encode(), value.encode()))

        # Always set correct content-length based on actual body
        if self.body:
            headers_list.append((b"content-length", str(len(self.body)).encode()))
        else:
            headers_list.append((b"content-length", b"0"))

        # Ensure we have content-type
        has_content_type = any(h[0] == b"content-type" for h in headers_list)
        if not has_content_type and self.media_type:
            headers_list.append((b"content-type", self.media_type.encode()))

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
