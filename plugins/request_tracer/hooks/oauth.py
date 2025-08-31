"""OAuth-specific request tracer hook."""

import json
from pathlib import Path
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
                "oauth_tracer_hook_initialized",
                json_logs=self.config.json_logs_enabled,
                raw_http=self.config.raw_http_enabled,
                log_dir=self.config.log_dir,
            )
    
    async def process(self, event: HookEvent, context: HookContext) -> HookContext:
        """Process OAuth events and log them.
        
        Args:
            event: The hook event
            context: Hook context with event data
            
        Returns:
            Unmodified context
        """
        if not self.enabled:
            return context
        
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
                event=event,
                error=str(e),
                exc_info=e,
            )
        
        return context
    
    async def _log_oauth_request(self, context: HookContext, is_refresh: bool = False) -> None:
        """Log an OAuth request.
        
        Args:
            context: Hook context with request data
            is_refresh: Whether this is a token refresh request
        """
        provider = context.get("provider", "unknown")
        endpoint = context.get("endpoint", "")
        method = context.get("method", "POST")
        headers = context.get("headers", {})
        body = context.get("body", {})
        is_json = context.get("is_json", False)
        
        # Generate a request ID for correlation
        import uuid
        request_id = str(uuid.uuid4())
        
        # Store request ID in context for response correlation
        context["request_id"] = request_id
        
        operation = "refresh" if is_refresh else "token_exchange"
        logger.debug(
            f"oauth_{operation}_request",
            request_id=request_id,
            provider=provider,
            endpoint=endpoint,
        )
        
        # Log with JSON formatter
        if self.json_formatter:
            await self.json_formatter.log_client_request(
                request_id=request_id,
                method=method,
                url=endpoint,
                headers=headers,
                body=json.dumps(body) if is_json else self._encode_form_data(body),
            )
        
        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP request
            raw_request = self._build_raw_http_request(
                method, endpoint, headers, body, is_json
            )
            await self.raw_formatter.log_raw_client_request(
                request_id=request_id,
                raw_data=raw_request,
            )
    
    async def _log_oauth_response(self, context: HookContext, is_refresh: bool = False) -> None:
        """Log an OAuth response.
        
        Args:
            context: Hook context with response data
            is_refresh: Whether this is a token refresh response
        """
        provider = context.get("provider", "unknown")
        endpoint = context.get("endpoint", "")
        status_code = context.get("status_code", 0)
        headers = context.get("headers", {})
        body = context.get("body", {})
        request_id = context.get("request_id", str(uuid.uuid4()))
        
        operation = "refresh" if is_refresh else "token_exchange"
        logger.debug(
            f"oauth_{operation}_response",
            request_id=request_id,
            provider=provider,
            status_code=status_code,
        )
        
        # Log with JSON formatter
        if self.json_formatter:
            await self.json_formatter.log_client_response(
                request_id=request_id,
                status_code=status_code,
                headers=headers,
                body=json.dumps(body) if body else None,
            )
        
        # Log with raw HTTP formatter
        if self.raw_formatter:
            # Build raw HTTP response
            raw_response = self._build_raw_http_response(
                status_code, headers, body
            )
            await self.raw_formatter.log_raw_client_response(
                request_id=request_id,
                raw_data=raw_response,
            )
    
    async def _log_oauth_error(self, context: HookContext) -> None:
        """Log an OAuth error.
        
        Args:
            context: Hook context with error data
        """
        provider = context.get("provider", "unknown")
        endpoint = context.get("endpoint", "")
        error_type = context.get("error_type", "unknown")
        status_code = context.get("status_code", 0)
        error_detail = context.get("error_detail", "")
        response_body = context.get("response_body", "")
        request_id = context.get("request_id", str(uuid.uuid4()))
        
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
            await self.json_formatter.log_client_response(
                request_id=request_id,
                status_code=status_code,
                headers={},
                body=response_body,
            )
        
        if self.raw_formatter and status_code > 0:
            # Build error response
            raw_response = f"HTTP/1.1 {status_code} Error\r\n\r\n{response_body}"
            await self.raw_formatter.log_raw_client_response(
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