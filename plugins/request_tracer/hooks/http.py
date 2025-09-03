"""Generic HTTP request tracer hook."""

import json
import uuid
from typing import Any

import structlog

from ccproxy.hooks import Hook
from ccproxy.hooks.base import HookContext
from ccproxy.hooks.events import HookEvent

from ..config import RequestTracerConfig
from ..formatters import JSONFormatter, RawHTTPFormatter


logger = structlog.get_logger(__name__)


class HTTPTracerHook(Hook):
    """Hook for tracing all HTTP requests and responses."""

    name = "http_tracer"
    events = [
        HookEvent.HTTP_REQUEST,
        HookEvent.HTTP_RESPONSE,
        HookEvent.HTTP_ERROR,
    ]
    priority = 100  # Run early to capture raw data

    def __init__(self, config: RequestTracerConfig | None = None) -> None:
        """Initialize the HTTP tracer hook.

        Args:
            config: Request tracer configuration
        """
        self.config = config or RequestTracerConfig()
        self.enabled = self.config.enabled

        # Initialize formatters if enabled
        self.json_formatter = (
            JSONFormatter(self.config) if self.config.json_logs_enabled else None
        )
        self.raw_formatter = (
            RawHTTPFormatter(self.config) if self.config.raw_http_enabled else None
        )

        if self.enabled:
            logger.info(
                "http_tracer_hook_initialized",
                json_logs=self.config.json_logs_enabled,
                raw_http=self.config.raw_http_enabled,
                log_dir=self.config.log_dir,
            )

    async def __call__(self, context: HookContext) -> None:
        """Process HTTP events and log them.

        Args:
            context: Hook context with event data
        """
        if not self.enabled:
            return

        event = context.event
        try:
            if event == HookEvent.HTTP_REQUEST:
                await self._log_http_request(context)
            elif event == HookEvent.HTTP_RESPONSE:
                await self._log_http_response(context)
            elif event == HookEvent.HTTP_ERROR:
                await self._log_http_error(context)
        except Exception as e:
            logger.error(
                "http_tracer_hook_error",
                hook_event=event.value if hasattr(event, "value") else str(event),
                error=str(e),
                exc_info=e,
            )

    async def _log_http_request(self, context: HookContext) -> None:
        """Log an HTTP request.

        Args:
            context: Hook context with request data
        """
        method = context.data.get("method", "UNKNOWN")
        url = context.data.get("url", "")
        headers = context.data.get("headers", {})
        body = context.data.get("body")
        is_json = context.data.get("is_json", False)

        # Generate a request ID for correlation
        request_id = str(uuid.uuid4())

        # Store request ID in context for response correlation
        context.data["request_id"] = request_id

        logger.debug(
            "http_request",
            request_id=request_id,
            method=method,
            url=url,
        )

        # Log with JSON formatter
        if self.json_formatter:
            # Pass body directly - JSONFormatter now handles different data types
            await self.json_formatter.log_request(
                request_id=request_id,
                method=method,
                url=url,
                headers=headers,
                body=body,  # Pass original body data directly
                request_type="http",
            )

        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP request
            raw_request = self._build_raw_http_request(
                method, url, headers, body, is_json
            )
            await self.raw_formatter.log_client_request(
                request_id=request_id,
                raw_data=raw_request,
            )

    async def _log_http_response(self, context: HookContext) -> None:
        """Log an HTTP response.

        Args:
            context: Hook context with response data
        """
        request_id = context.data.get("request_id", str(uuid.uuid4()))
        status_code = context.data.get("status_code", 0)
        headers = context.data.get("response_headers", {})
        body_any = context.data.get("response_body")

        logger.debug(
            "http_response",
            request_id=request_id,
            status_code=status_code,
        )

        # Log with JSON formatter
        if self.json_formatter:
            # Normalize body to bytes for formatter typing
            if body_any is None:
                body_bytes = b""
            elif isinstance(body_any, bytes):
                body_bytes = body_any
            elif isinstance(body_any, str):
                body_bytes = body_any.encode("utf-8")
            else:
                body_bytes = json.dumps(body_any).encode("utf-8")

            await self.json_formatter.log_response(
                request_id=request_id,
                status=status_code,
                headers=headers,
                body=body_bytes,
                response_type="http",
            )

        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP response
            raw_response = self._build_raw_http_response(status_code, headers, body_any)
            await self.raw_formatter.log_client_response(
                request_id=request_id,
                raw_data=raw_response,
            )

    async def _log_http_error(self, context: HookContext) -> None:
        """Log an HTTP error.

        Args:
            context: Hook context with error data
        """
        request_id = context.data.get("request_id", str(uuid.uuid4()))
        error_type = context.data.get("error_type", "unknown")
        error_detail = context.data.get("error_detail", "")
        status_code = context.data.get("status_code", 0)
        response_body = context.data.get("response_body", "")

        logger.error(
            "http_error",
            request_id=request_id,
            error_type=error_type,
            status_code=status_code,
            error_detail=error_detail,
        )

        # Log error response with formatters
        if self.json_formatter:
            await self.json_formatter.log_error(
                request_id=request_id,
                error=Exception(f"{error_type}: {error_detail}"),
            )

        if self.raw_formatter and status_code > 0:
            # Build error response
            raw_response = f"HTTP/1.1 {status_code} Error\r\n\r\n{response_body}"
            await self.raw_formatter.log_client_response(
                request_id=request_id,
                raw_data=raw_response.encode(),
            )

    def _build_raw_http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any,
        is_json: bool,
    ) -> bytes:
        """Build raw HTTP request for logging.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            body: Request body
            is_json: Whether body is JSON

        Returns:
            Raw HTTP request bytes
        """
        # Parse URL to get path
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        # Build request line
        lines = [f"{method} {path} HTTP/1.1"]

        # Add Host header
        if parsed.netloc:
            lines.append(f"Host: {parsed.netloc}")

        # Add other headers
        for key, value in headers.items():
            lines.append(f"{key}: {value}")

        # Add body
        body_str = ""
        if body:
            if is_json and isinstance(body, dict):
                body_str = json.dumps(body)
            elif isinstance(body, bytes):
                try:
                    body_str = body.decode()
                except (UnicodeDecodeError, AttributeError):
                    body_str = str(body)
            else:
                body_str = str(body)

            lines.append(f"Content-Length: {len(body_str)}")
            lines.append("")
            lines.append(body_str)
        else:
            lines.append("")

        return "\r\n".join(lines).encode()

    def _build_raw_http_response(
        self,
        status_code: int,
        headers: dict[str, str],
        body: Any,
    ) -> bytes:
        """Build raw HTTP response for logging.

        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body

        Returns:
            Raw HTTP response bytes
        """
        # Build status line
        lines = [f"HTTP/1.1 {status_code} OK"]

        # Add headers
        for key, value in headers.items():
            lines.append(f"{key}: {value}")

        # Add body
        if body:
            if isinstance(body, dict):
                body_str = json.dumps(body, indent=2)
            elif isinstance(body, bytes):
                try:
                    body_str = body.decode()
                except (UnicodeDecodeError, AttributeError):
                    body_str = str(body)
            else:
                body_str = str(body)

            lines.append(f"Content-Length: {len(body_str)}")
            lines.append("")
            lines.append(body_str)
        else:
            lines.append("")

        return "\r\n".join(lines).encode()
