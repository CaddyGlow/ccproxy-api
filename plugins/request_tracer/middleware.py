"""ASGI middleware for request tracing."""

import time
from collections.abc import Callable
from typing import Any

from ccproxy.observability import (
    ClientRequestEvent,
    ClientResponseEvent,
    get_observability_pipeline,
)
from ccproxy.observability.context import RequestContext

from .tracer import RequestTracerImpl


class RequestTracingMiddleware:
    """ASGI middleware for tracing HTTP requests and responses."""

    def __init__(
        self, app: Callable[..., Any], tracer: RequestTracerImpl | None = None
    ) -> None:
        self.app = app
        self.tracer = tracer
        self.pipeline = get_observability_pipeline()

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

        # Skip if tracing is disabled or tracer not set
        if not self.tracer or not self.tracer.should_log_raw():
            await self.app(scope, receive, send)
            return

        # Check if path should be traced based on include/exclude rules
        path = scope.get("path", "/")
        if not self.tracer.should_trace_path(path):
            await self.app(scope, receive, send)
            return

        # Extract request ID from headers or generate one
        request_id = self._get_request_id(scope)

        # Buffer to collect request body
        request_body_chunks = []

        # Wrap receive to capture request body chunks
        async def wrapped_receive() -> dict[str, Any]:
            message: dict[str, Any] = await receive()

            # Capture request body chunks
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    request_body_chunks.append(body)
                    # Log for raw HTTP tracing
                    if self.tracer:
                        await self.tracer.log_raw_client_request(request_id, body)

                # If this is the last chunk, emit the request event with full body
                more_body = message.get("more_body", False)
                if not more_body:
                    full_body = (
                        b"".join(request_body_chunks) if request_body_chunks else None
                    )
                    # Emit client request event with body
                    await self._emit_client_request_event(scope, request_id, full_body)

            return message

        # Log initial request headers (for raw tracing if enabled)
        await self._log_request_headers(scope, request_id)

        # Track request start time for response event
        request_start_time = time.time()

        # Wrap send to capture response chunks and emit response event
        wrapped_send = self._wrap_send(send, request_id, request_start_time)

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

    async def _emit_client_request_event(
        self, scope: dict[str, Any], request_id: str, body: bytes | None = None
    ) -> None:
        """Emit client request event to the observability pipeline."""
        # Get current RequestContext if available
        context = RequestContext.get_current()

        # Extract headers from scope
        headers_dict = {}
        for name, value in scope.get("headers", []):
            headers_dict[name.decode("utf-8", errors="ignore")] = value.decode(
                "utf-8", errors="ignore"
            )

        # If we have context, prefer its metadata; otherwise extract from scope
        if context:
            method = context.metadata.get("method", scope.get("method", "GET"))
            path = context.metadata.get("path", scope.get("path", "/"))
            query = context.metadata.get("query")
            client_ip = context.metadata.get("client_ip")
            user_agent = context.metadata.get("user_agent")
        else:
            # Fallback to manual extraction
            method = scope.get("method", "GET")
            path = scope.get("path", "/")
            query_string = scope.get("query_string", b"")
            query = query_string.decode("utf-8") if query_string else None

            # Extract client IP and user agent from headers
            client_ip = None
            user_agent = headers_dict.get("user-agent")

            # Check for forwarded IPs
            if "x-forwarded-for" in headers_dict:
                client_ip = headers_dict["x-forwarded-for"].split(",")[0].strip()
            elif "x-real-ip" in headers_dict:
                client_ip = headers_dict["x-real-ip"]

            # Fallback to client info from scope
            if not client_ip:
                client_info = scope.get("client")
                if client_info:
                    client_ip = (
                        client_info[0]
                        if isinstance(client_info, tuple | list)
                        else str(client_info)
                    )

        event = ClientRequestEvent(
            request_id=request_id,
            method=method,
            path=path,
            query=query,
            headers=headers_dict,  # Include headers!
            body=body,  # Include the body!
            client_ip=client_ip,
            user_agent=user_agent,
            context=context,  # Pass the full context!
        )

        await self.pipeline.notify_client_request(event)

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
        if self.tracer and hasattr(self.tracer.config, "exclude_headers"):
            exclude_headers = [
                h.lower().encode() for h in self.tracer.config.exclude_headers
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

        if self.tracer:
            await self.tracer.log_raw_client_request(request_id, raw)

    def _wrap_send(
        self,
        send: Callable[[dict[str, Any]], Any],
        request_id: str,
        request_start_time: float,
    ) -> Callable[[dict[str, Any]], Any]:
        """Wrap send to capture response chunks."""
        logged_headers = False
        response_status = 200
        response_body_size = 0
        response_headers = {}
        response_body_chunks: list[bytes] = []  # Buffer for accumulating body

        async def wrapped(message: dict[str, Any]) -> None:
            nonlocal \
                logged_headers, \
                response_status, \
                response_body_size, \
                response_headers, \
                response_body_chunks

            if message["type"] == "http.response.start":
                # Capture status for event
                response_status = message.get("status", 200)
                headers = message.get("headers", [])

                # Convert headers to dict for event
                for name, value in headers:
                    response_headers[name.decode("utf-8", errors="ignore")] = (
                        value.decode("utf-8", errors="ignore")
                    )

                # Log response headers for raw tracing if enabled
                if self.tracer and self.tracer.should_log_raw():
                    # Build raw response headers
                    lines = [f"HTTP/1.1 {response_status} OK"]

                    # Add headers (with optional filtering)
                    exclude_headers = []
                    if hasattr(self.tracer.config, "exclude_headers"):
                        exclude_headers = [
                            h.lower().encode()
                            for h in self.tracer.config.exclude_headers
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

                    await self.tracer.log_raw_client_response(request_id, raw)
                logged_headers = True

            elif message["type"] == "http.response.body":
                # Track body size and accumulate chunks
                body = message.get("body", b"")
                if body:
                    response_body_size += len(body)
                    response_body_chunks.append(body)  # Accumulate chunks

                    # Log response body chunks for raw tracing if enabled
                    if self.tracer and self.tracer.should_log_raw():
                        await self.tracer.log_raw_client_response(request_id, body)

                # If this is the final chunk, emit response event with full body
                more_body = message.get("more_body", False)
                if not more_body:
                    # Calculate duration
                    duration_ms = (time.time() - request_start_time) * 1000

                    # Get current context
                    context = RequestContext.get_current()

                    # Combine all body chunks
                    full_body = (
                        b"".join(response_body_chunks) if response_body_chunks else None
                    )

                    # Emit client response event with full body
                    event = ClientResponseEvent(
                        request_id=request_id,
                        status_code=response_status,
                        headers=response_headers,  # Include headers!
                        body=full_body,  # Include the full accumulated body!
                        body_size=response_body_size,
                        duration_ms=duration_ms,
                        context=context,  # Pass the context!
                    )
                    await self.pipeline.notify_client_response(event)

            # Forward message
            await send(message)

        return wrapped
