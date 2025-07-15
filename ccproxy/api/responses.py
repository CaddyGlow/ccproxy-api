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

        # Add all headers from the response
        for name, value in self._preserve_headers.items():
            headers_list.append((name.lower().encode(), value.encode()))

        # Ensure we have proper content-type and content-length
        has_content_type = False
        has_content_length = False

        for header in headers_list:
            if header[0] == b"content-type":
                has_content_type = True
            elif header[0] == b"content-length":
                has_content_length = True

        # Add missing headers if needed
        if not has_content_type and self.media_type:
            headers_list.append((b"content-type", self.media_type.encode()))
        if not has_content_length and self.body:
            headers_list.append((b"content-length", str(len(self.body)).encode()))

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
