"""ASGI middleware for logging raw HTTP data from client requests."""

from collections.abc import Callable
from typing import Any

from .logger import RawHTTPLogger


class RawHTTPLoggingMiddleware:
    """ASGI middleware to log raw HTTP data without buffering."""

    def __init__(
        self, app: Callable[..., Any], logger: RawHTTPLogger | None = None
    ) -> None:
        self.app = app
        self.logger = logger or RawHTTPLogger()

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Process ASGI request with raw logging."""
        # Only handle HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip if logging is disabled
        if not self.logger.should_log():
            await self.app(scope, receive, send)
            return

        # Check if path should be logged based on include/exclude rules
        path = scope.get("path", "/")
        if hasattr(self.logger, "config") and self.logger.config:
            # First check exclude_paths (takes precedence)
            if any(
                path.startswith(exclude) for exclude in self.logger.config.exclude_paths
            ):
                await self.app(scope, receive, send)
                return

            # Then check include_paths (if specified, only log included paths)
            if self.logger.config.include_paths and not any(
                path.startswith(include) for include in self.logger.config.include_paths
            ):
                await self.app(scope, receive, send)
                return

        # Extract request ID from headers or generate one
        request_id = self._get_request_id(scope)

        # Log initial request headers
        await self._log_request_headers(scope, request_id)

        # Wrap receive to capture request body chunks
        wrapped_receive = self._wrap_receive(receive, request_id)

        # Wrap send to capture response chunks
        wrapped_send = self._wrap_send(send, request_id)

        # Forward to app
        await self.app(scope, wrapped_receive, wrapped_send)

    def _get_request_id(self, scope: dict[str, Any]) -> str:
        """Extract request ID from ASGI scope or headers."""
        # First check ASGI extensions (set by RequestIDMiddleware)
        if "extensions" in scope and "request_id" in scope["extensions"]:
            return str(scope["extensions"]["request_id"])

        # Fallback: Look for request ID in headers
        headers = dict(scope.get("headers", []))
        for header_name in [b"x-request-id", b"x-correlation-id"]:
            if header_name in headers:
                return str(headers[header_name].decode("utf-8"))

        # Last resort: Generate a UUID (consistent with RequestIDMiddleware)
        import uuid

        return str(uuid.uuid4())

    async def _log_request_headers(
        self, scope: dict[str, Any], request_id: str
    ) -> None:
        """Log the initial request headers."""
        # Build raw request line
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        query_string = scope.get("query_string", b"")

        if query_string:
            full_path = f"{path}?{query_string.decode('utf-8')}"
        else:
            full_path = path

        # Build raw request headers
        lines = [f"{method} {full_path} HTTP/1.1"]

        # Add headers (with optional filtering)
        exclude_headers = []
        if hasattr(self.logger, "config") and self.logger.config:
            exclude_headers = [
                h.lower().encode() for h in self.logger.config.exclude_headers
            ]

        for name, value in scope.get("headers", []):
            if name.lower() not in exclude_headers:
                lines.append(
                    f"{name.decode('ascii')}: {value.decode('ascii', errors='ignore')}"
                )
            else:
                lines.append(f"{name.decode('ascii')}: [REDACTED]")

        # Build raw request
        raw = "\r\n".join(lines).encode("utf-8")
        raw += b"\r\n\r\n"

        await self.logger.log_client_request(request_id, raw)

    def _wrap_receive(
        self, receive: Callable[[], Any], request_id: str
    ) -> Callable[[], Any]:
        """Wrap receive to capture request body chunks."""

        async def wrapped() -> dict[str, Any]:
            message: dict[str, Any] = await receive()

            # Log request body chunks
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    await self.logger.log_client_request(request_id, body)

            return message

        return wrapped

    def _wrap_send(
        self, send: Callable[[dict[str, Any]], Any], request_id: str
    ) -> Callable[[dict[str, Any]], Any]:
        """Wrap send to capture response chunks."""
        logged_headers = False

        async def wrapped(message: dict[str, Any]) -> None:
            nonlocal logged_headers

            if message["type"] == "http.response.start":
                # Log response headers
                status = message.get("status", 200)
                headers = message.get("headers", [])

                # Build raw response headers
                lines = [f"HTTP/1.1 {status} OK"]

                # Add headers (with optional filtering)
                exclude_headers = []
                if hasattr(self.logger, "config") and self.logger.config:
                    exclude_headers = [
                        h.lower().encode() for h in self.logger.config.exclude_headers
                    ]

                for name, value in headers:
                    if name.lower() not in exclude_headers:
                        lines.append(
                            f"{name.decode('ascii')}: {value.decode('ascii', errors='ignore')}"
                        )
                    else:
                        lines.append(f"{name.decode('ascii')}: [REDACTED]")

                raw = "\r\n".join(lines).encode("utf-8")
                raw += b"\r\n\r\n"

                await self.logger.log_client_response(request_id, raw)
                logged_headers = True

            elif message["type"] == "http.response.body":
                # Log response body chunks
                body = message.get("body", b"")
                if body:
                    await self.logger.log_client_response(request_id, body)

            # Forward message
            await send(message)

        return wrapped
