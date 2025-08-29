"""Hook-based access log implementation."""

import time
from typing import Any

import structlog

from ccproxy.hooks import Hook
from ccproxy.hooks.base import HookContext
from ccproxy.hooks.events import HookEvent

from .config import AccessLogConfig
from .formatter import AccessLogFormatter
from .writer import AccessLogWriter


logger = structlog.get_logger(__name__)


class AccessLogHook(Hook):
    """Hook-based access logger implementation.
    
    This hook listens to request/response lifecycle events and logs them
    according to the configured format (common, combined, or structured).
    """
    
    name = "access_log"
    events = [
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.REQUEST_FAILED,
        HookEvent.PROVIDER_REQUEST_SENT,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_ERROR,
    ]
    
    def __init__(self, config: AccessLogConfig | None = None) -> None:
        """Initialize the access log hook.
        
        Args:
            config: Access log configuration
        """
        self.config = config or AccessLogConfig()
        self.formatter = AccessLogFormatter()
        
        # Create writers based on configuration
        self.client_writer: AccessLogWriter | None = None
        self.provider_writer: AccessLogWriter | None = None
        
        if self.config.client_enabled:
            self.client_writer = AccessLogWriter(
                self.config.client_log_file,
                self.config.buffer_size,
                self.config.flush_interval,
            )
        
        if self.config.provider_enabled:
            self.provider_writer = AccessLogWriter(
                self.config.provider_log_file,
                self.config.buffer_size,
                self.config.flush_interval,
            )
        
        # Track in-flight requests
        self.client_requests: dict[str, dict[str, Any]] = {}
        self.provider_requests: dict[str, dict[str, Any]] = {}
        
        logger.info(
            "access_log_hook_initialized",
            enabled=self.config.enabled,
            client_enabled=self.config.client_enabled,
            client_format=self.config.client_format,
            provider_enabled=self.config.provider_enabled,
        )
    
    async def __call__(self, context: HookContext) -> None:
        """Handle hook events for access logging.
        
        Args:
            context: Hook context with event data
        """
        if not self.config.enabled:
            return
        
        # Map hook events to handler methods
        handlers = {
            HookEvent.REQUEST_STARTED: self._handle_request_start,
            HookEvent.REQUEST_COMPLETED: self._handle_request_complete,
            HookEvent.REQUEST_FAILED: self._handle_request_failed,
            HookEvent.PROVIDER_REQUEST_SENT: self._handle_provider_request,
            HookEvent.PROVIDER_RESPONSE_RECEIVED: self._handle_provider_response,
            HookEvent.PROVIDER_ERROR: self._handle_provider_error,
        }
        
        handler = handlers.get(context.event)
        if handler:
            try:
                await handler(context)
            except Exception as e:
                logger.error(
                    "access_log_hook_error",
                    hook_event=context.event.value if context.event else "unknown",
                    error=str(e),
                    exc_info=e,
                )
    
    async def _handle_request_start(self, context: HookContext) -> None:
        """Handle REQUEST_STARTED event."""
        if not self.config.client_enabled:
            return
        
        # Extract request data from context
        request_id = context.data.get("request_id", "unknown")
        method = context.data.get("method", "UNKNOWN")
        
        # Handle both path and url fields
        path = context.data.get("path", "")
        if not path and "url" in context.data:
            # Extract path from URL
            url = context.data.get("url", "")
            path = self._extract_path(url)
        
        query = context.data.get("query", "")
        
        # Try to get client_ip from various sources
        client_ip = context.data.get("client_ip", "-")
        if client_ip == "-" and context.request:
            # Try to get from request object
            if hasattr(context.request, 'client'):
                client_ip = getattr(context.request.client, 'host', "-") if context.request.client else "-"
        
        # Try to get user_agent from headers
        user_agent = context.data.get("user_agent", "-")
        if user_agent == "-":
            headers = context.data.get("headers", {})
            user_agent = headers.get("user-agent", "-")
        
        # Check path filters
        if self._should_exclude_path(path):
            return
        
        # Store request data for later
        # Get current time for timestamp
        current_time = time.time()
            
        self.client_requests[request_id] = {
            "timestamp": current_time,  # Store as float for formatter compatibility
            "method": method,
            "path": path,
            "query": query,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "start_time": current_time,
        }
    
    async def _handle_request_complete(self, context: HookContext) -> None:
        """Handle REQUEST_COMPLETED event."""
        if not self.config.client_enabled:
            return
        
        request_id = context.data.get("request_id", "unknown")
        
        # Check if we have the request data
        if request_id not in self.client_requests:
            return
        
        # Get and remove request data
        request_data = self.client_requests.pop(request_id)
        
        # Calculate duration
        duration_ms = (time.time() - request_data["start_time"]) * 1000
        
        # Extract response data
        status_code = context.data.get("status_code", 200)
        body_size = context.data.get("body_size", 0)
        
        # Merge request and response data
        log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": status_code,
            "body_size": body_size,
            "duration_ms": duration_ms,
            "error": None,
        }
        
        # Format and write
        if self.client_writer:
            formatted = self.formatter.format_client(
                log_data, self.config.client_format
            )
            await self.client_writer.write(formatted)
        
        # Also log to structured logger
        await self._log_to_structured_logger(log_data, "client")
    
    async def _handle_request_failed(self, context: HookContext) -> None:
        """Handle REQUEST_FAILED event."""
        if not self.config.client_enabled:
            return
        
        request_id = context.data.get("request_id", "unknown")
        
        # Check if we have the request data
        if request_id not in self.client_requests:
            return
        
        # Get and remove request data
        request_data = self.client_requests.pop(request_id)
        
        # Calculate duration
        duration_ms = (time.time() - request_data["start_time"]) * 1000
        
        # Extract error information
        error = context.error
        error_message = str(error) if error else "Unknown error"
        status_code = context.data.get("status_code", 500)
        
        # Merge request and error data
        log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": status_code,
            "body_size": 0,
            "duration_ms": duration_ms,
            "error": error_message,
        }
        
        # Format and write
        if self.client_writer:
            formatted = self.formatter.format_client(
                log_data, self.config.client_format
            )
            await self.client_writer.write(formatted)
        
        # Also log to structured logger
        await self._log_to_structured_logger(log_data, "client", error=error_message)
    
    async def _handle_provider_request(self, context: HookContext) -> None:
        """Handle PROVIDER_REQUEST_SENT event."""
        if not self.config.provider_enabled:
            return
        
        request_id = context.metadata.get("request_id", "unknown")
        provider = context.provider or "unknown"
        url = context.data.get("url", "")
        method = context.data.get("method", "UNKNOWN")
        
        # Store request data for later
        # Get current time for timestamp
        current_time = time.time()
            
        self.provider_requests[request_id] = {
            "timestamp": current_time,  # Store as float for formatter compatibility
            "provider": provider,
            "method": method,
            "url": url,
            "start_time": current_time,
        }
    
    async def _handle_provider_response(self, context: HookContext) -> None:
        """Handle PROVIDER_RESPONSE_RECEIVED event."""
        if not self.config.provider_enabled:
            return
        
        request_id = context.metadata.get("request_id", "unknown")
        
        # Check if we have the request data
        if request_id not in self.provider_requests:
            return
        
        # Get and remove request data
        request_data = self.provider_requests.pop(request_id)
        
        # Calculate duration if not provided
        duration_ms = context.data.get("duration_ms", 0)
        if duration_ms == 0:
            duration_ms = (time.time() - request_data["start_time"]) * 1000
        
        # Extract response data
        status_code = context.data.get("status_code", 200)
        tokens_input = context.data.get("tokens_input", 0)
        tokens_output = context.data.get("tokens_output", 0)
        cache_read_tokens = context.data.get("cache_read_tokens", 0)
        cache_write_tokens = context.data.get("cache_write_tokens", 0)
        cost_usd = context.data.get("cost_usd", 0.0)
        model = context.data.get("model", "")
        
        # Merge request and response data
        log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": cost_usd,
            "model": model,
        }
        
        # Format and write
        if self.provider_writer:
            formatted = self.formatter.format_provider(log_data)
            await self.provider_writer.write(formatted)
        
        # Also log to structured logger
        await self._log_to_structured_logger(log_data, "provider")
    
    async def _handle_provider_error(self, context: HookContext) -> None:
        """Handle PROVIDER_ERROR event."""
        if not self.config.provider_enabled:
            return
        
        request_id = context.metadata.get("request_id", "unknown")
        
        # Check if we have the request data
        if request_id not in self.provider_requests:
            return
        
        # Get and remove request data
        request_data = self.provider_requests.pop(request_id)
        
        # Calculate duration
        duration_ms = (time.time() - request_data["start_time"]) * 1000
        
        # Extract error information
        error = context.error
        error_message = str(error) if error else "Unknown error"
        status_code = context.data.get("status_code", 500)
        
        # Merge request and error data
        log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "tokens_input": 0,
            "tokens_output": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
            "model": "",
            "error": error_message,
        }
        
        # Format and write
        if self.provider_writer:
            formatted = self.formatter.format_provider(log_data)
            await self.provider_writer.write(formatted)
        
        # Also log to structured logger
        await self._log_to_structured_logger(log_data, "provider", error=error_message)
    
    def _extract_path(self, url: str) -> str:
        """Extract path from URL.
        
        Args:
            url: Full URL or path
            
        Returns:
            The path portion of the URL
        """
        if "://" in url:
            # Full URL - extract path
            parts = url.split("/", 3)
            return "/" + parts[3] if len(parts) > 3 else "/"
        return url
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if a path should be excluded from logging.
        
        Args:
            path: The request path
            
        Returns:
            True if the path should be excluded, False otherwise
        """
        for excluded in self.config.exclude_paths:
            if path.startswith(excluded):
                return True
        return False
    
    async def _log_to_structured_logger(
        self,
        log_data: dict[str, Any],
        log_type: str,
        error: str | None = None,
    ) -> None:
        """Log to structured logger (stdout/stderr).
        
        Args:
            log_data: Log data dictionary
            log_type: Type of log ("client" or "provider")
            error: Error message if applicable
        """
        # Prepare structured log entry
        structured_data = {
            "log_type": log_type,
            "request_id": log_data.get("request_id"),
            "method": log_data.get("method"),
            "path": log_data.get("path"),
            "status_code": log_data.get("status_code"),
            "duration_ms": log_data.get("duration_ms"),
            "client_ip": log_data.get("client_ip"),
            "user_agent": log_data.get("user_agent"),
        }
        
        # Add provider-specific fields
        if log_type == "provider":
            structured_data.update({
                "provider": log_data.get("provider"),
                "url": log_data.get("url"),
                "tokens_input": log_data.get("tokens_input"),
                "tokens_output": log_data.get("tokens_output"),
                "cache_read_tokens": log_data.get("cache_read_tokens"),
                "cache_write_tokens": log_data.get("cache_write_tokens"),
                "cost_usd": log_data.get("cost_usd"),
                "model": log_data.get("model"),
            })
        
        # Remove None values to keep log clean
        structured_data = {k: v for k, v in structured_data.items() if v is not None}
        
        # Log with appropriate level - event is passed as first argument to logger methods
        if error:
            logger.warning("access_log", error=error, **structured_data)
        else:
            logger.info("access_log", **structured_data)
    
    async def close(self) -> None:
        """Close writers and flush any pending data."""
        if self.client_writer:
            await self.client_writer.close()
        if self.provider_writer:
            await self.provider_writer.close()