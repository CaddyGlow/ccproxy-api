"""OAuth-specific request tracer hook."""

import json
from typing import Any

import structlog

from ccproxy.hooks import Hook
from ccproxy.hooks.base import HookContext
from ccproxy.hooks.events import HookEvent

from ..config import RequestTracerConfig
from ..formatters import JSONFormatter, RawHTTPFormatter


logger = structlog.get_logger(__name__)


class OAuthTracerHook(Hook):
    """Hook for tracing OAuth HTTP requests and responses."""

    name = "oauth_tracer"
    events = [
        HookEvent.OAUTH_TOKEN_REQUEST,
        HookEvent.OAUTH_TOKEN_RESPONSE,
        HookEvent.OAUTH_REFRESH_REQUEST,
        HookEvent.OAUTH_REFRESH_RESPONSE,
        HookEvent.OAUTH_ERROR,
    ]
    priority = 100  # Run early to capture raw data

    def __init__(self, config: RequestTracerConfig | None = None) -> None:
        """Initialize the OAuth tracer hook.

        Args:
            config: Request tracer configuration
        """
        self.config = config or RequestTracerConfig()
        self.enabled = self.config.enabled and getattr(self.config, "trace_oauth", True)

        # Initialize formatters if enabled
        self.json_formatter = (
            JSONFormatter(self.config) if self.config.json_logs_enabled else None
        )
        self.raw_formatter = (
            RawHTTPFormatter(self.config) if self.config.raw_http_enabled else None
        )

        if self.enabled:
            logger.info(
                "oauth_tracer_hook_initialized",
                json_logs=self.config.json_logs_enabled,
                raw_http=self.config.raw_http_enabled,
                log_dir=self.config.log_dir,
            )

    async def __call__(self, context: HookContext) -> None:
        """Process OAuth events and log them.

        Args:
            context: Hook context with event data
        """
        if not self.enabled:
            return

        event = context.event
        try:
            if event == HookEvent.OAUTH_TOKEN_REQUEST:
                await self._log_oauth_request(context)
            elif event == HookEvent.OAUTH_TOKEN_RESPONSE:
                await self._log_oauth_response(context)
            elif event == HookEvent.OAUTH_REFRESH_REQUEST:
                await self._log_oauth_request(context, is_refresh=True)
            elif event == HookEvent.OAUTH_REFRESH_RESPONSE:
                await self._log_oauth_response(context, is_refresh=True)
            elif event == HookEvent.OAUTH_ERROR:
                await self._log_oauth_error(context)
        except Exception as e:
            logger.error(
                "oauth_tracer_hook_error",
                hook_event=event.value if hasattr(event, "value") else str(event),
                error=str(e),
                exc_info=e,
            )

    async def _log_oauth_request(
        self, context: HookContext, is_refresh: bool = False
    ) -> None:
        """Log an OAuth request.

        Args:
            context: Hook context with request data
            is_refresh: Whether this is a token refresh request
        """
        provider = context.data.get("provider", "unknown")
        endpoint = context.data.get("endpoint", "")
        method = context.data.get("method", "POST")
        headers = context.data.get("headers", {})
        body = context.data.get("body", {})
        is_json = context.data.get("is_json", False)

        # Generate a request ID for correlation
        import uuid

        request_id = str(uuid.uuid4())

        # Store request ID in context for response correlation
        context.data["request_id"] = request_id

        operation = "refresh" if is_refresh else "token_exchange"
        logger.debug(
            f"oauth_{operation}_request",
            request_id=request_id,
            provider=provider,
            endpoint=endpoint,
        )

        # Log with JSON formatter
        if self.json_formatter:
            # Pass body directly - JSONFormatter now handles different data types
            # For form data, convert to string for better logging
            log_body = body
            if not is_json and body:
                log_body = self._encode_form_data(body)

            await self.json_formatter.log_request(
                request_id=request_id,
                method=method,
                url=endpoint,
                headers=headers,
                body=log_body,  # Pass original or form-encoded body directly
                request_type="oauth",
            )

        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP request
            raw_request = self._build_raw_http_request(
                method, endpoint, headers, body, is_json
            )
            await self.raw_formatter.log_client_request(
                request_id=request_id,
                raw_data=raw_request,
            )

    async def _log_oauth_response(
        self, context: HookContext, is_refresh: bool = False
    ) -> None:
        """Log an OAuth response.

        Args:
            context: Hook context with response data
            is_refresh: Whether this is a token refresh response
        """
        import uuid

        provider = context.data.get("provider", "unknown")
        endpoint = context.data.get("endpoint", "")
        status_code = context.data.get("status_code", 0)
        headers = context.data.get("headers", {})
        body = context.data.get("body", {})
        request_id = context.data.get("request_id", str(uuid.uuid4()))

        operation = "refresh" if is_refresh else "token_exchange"
        logger.debug(
            f"oauth_{operation}_response",
            request_id=request_id,
            provider=provider,
            status_code=status_code,
        )

        # Log with JSON formatter
        if self.json_formatter:
            # Pass body directly - JSONFormatter now handles different data types
            await self.json_formatter.log_response(
                request_id=request_id,
                status=status_code,
                headers=headers,
                body=body,  # Pass original body data directly
                response_type="oauth",
            )

        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP response
            raw_response = self._build_raw_http_response(status_code, headers, body)
            await self.raw_formatter.log_client_response(
                request_id=request_id,
                raw_data=raw_response,
            )

    async def _log_oauth_error(self, context: HookContext) -> None:
        """Log an OAuth error.

        Args:
            context: Hook context with error data
        """
        import uuid

        provider = context.data.get("provider", "unknown")
        endpoint = context.data.get("endpoint", "")
        error_type = context.data.get("error_type", "unknown")
        status_code = context.data.get("status_code", 0)
        error_detail = context.data.get("error_detail", "")
        response_body = context.data.get("response_body", "")
        request_id = context.data.get("request_id", str(uuid.uuid4()))

        logger.error(
            "oauth_error",
            request_id=request_id,
            provider=provider,
            endpoint=endpoint,
            error_type=error_type,
            status_code=status_code,
            error_detail=error_detail,
        )

        # Log error response with formatters
        if self.json_formatter:
            await self.json_formatter.log_error(
                request_id=request_id,
                error=Exception(f"{error_type}: {error_detail}"),
                provider=provider,
            )

        if self.raw_formatter and status_code > 0:
            # Build error response
            raw_response = f"HTTP/1.1 {status_code} Error\r\n\r\n{response_body}"
            await self.raw_formatter.log_client_response(
                request_id=request_id,
                raw_data=raw_response.encode(),
            )

    def _encode_form_data(self, data: dict[str, Any]) -> str:
        """Encode form data for logging.

        Args:
            data: Form data dictionary

        Returns:
            URL-encoded form data string
        """
        from urllib.parse import urlencode

        return urlencode(data)

    def _build_raw_http_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
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
        path = parsed.path + ("?" + parsed.query if parsed.query else "")

        # Build request line
        lines = [f"{method} {path} HTTP/1.1"]

        # Add Host header
        lines.append(f"Host: {parsed.netloc}")

        # Add other headers
        for key, value in headers.items():
            lines.append(f"{key}: {value}")

        # Add body
        if body:
            if is_json:
                body_str = json.dumps(body)
                lines.append(f"Content-Length: {len(body_str)}")
                lines.append("")
                lines.append(body_str)
            else:
                body_str = self._encode_form_data(body)
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
        body: dict[str, Any] | None,
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
            body_str = json.dumps(body, indent=2)
            lines.append(f"Content-Length: {len(body_str)}")
            lines.append("")
            lines.append(body_str)
        else:
            lines.append("")

        return "\r\n".join(lines).encode()
