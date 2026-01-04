"""Hook-based access log implementation."""

import time
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent

from .config import AccessLogConfig
from .formatter import AccessLogFormatter
from .writer import AccessLogWriter


logger = get_plugin_logger(__name__)


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
        HookEvent.PROVIDER_REQUEST_PREPARED,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_ERROR,
        HookEvent.PROVIDER_STREAM_END,
    ]
    priority = (
        750  # HookLayer.OBSERVATION + 50 - Access logging last to capture all data
    )

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
        # Store streaming metrics until REQUEST_COMPLETED fires
        self._streaming_metrics: dict[str, dict[str, Any]] = {}

        self.ingest_service: Any | None = None

        logger.trace(
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
            HookEvent.PROVIDER_REQUEST_PREPARED: self._handle_provider_request,
            HookEvent.PROVIDER_RESPONSE_RECEIVED: self._handle_provider_response,
            HookEvent.PROVIDER_ERROR: self._handle_provider_error,
            HookEvent.PROVIDER_STREAM_END: self._handle_provider_stream_end,
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
        if client_ip == "-" and context.request and hasattr(context.request, "client"):
            # Try to get from request object
            client_ip = (
                getattr(context.request.client, "host", "-")
                if context.request.client
                else "-"
            )

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

        # Store request data with additional context fields
        request_data = {
            "timestamp": current_time,  # Store as float for formatter compatibility
            "method": method,
            "path": path,
            "query": query,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "start_time": current_time,
        }

        # Add additional context fields if available
        additional_fields = [
            "endpoint",
            "service_type",
            "provider",
            "model",
            "session_id",
            "session_type",
            "streaming",
        ]
        for field in additional_fields:
            value = context.data.get(field)
            if value is not None:
                request_data[field] = value

        self.client_requests[request_id] = request_data

    async def _handle_request_complete(self, context: HookContext) -> None:
        """Handle REQUEST_COMPLETED event."""
        if not self.config.client_enabled:
            return

        request_id = context.data.get("request_id", "unknown")

        # Check if we have the request data
        if request_id not in self.client_requests:
            return

        # Check if this is a streaming response by looking for streaming flag
        # For streaming responses, we'll handle logging in PROVIDER_STREAM_END
        # to ensure we have all metrics
        is_streaming = (
            context.data.get("streaming_completed", False)
            or context.data.get("streaming", False)
            or self.client_requests.get(request_id, {}).get("streaming", False)
        )

        if is_streaming:
            # Check if we have metrics in metadata (non-streaming response wrapped as streaming)
            has_metrics = False
            if context.metadata:
                # Check if we have token metrics available
                has_metrics = any(
                    context.metadata.get(field) is not None
                    for field in ["tokens_input", "tokens_output", "cost_usd"]
                )

            if not has_metrics:
                # True streaming response - wait for PROVIDER_STREAM_END
                # Just mark that we got the completion
                if request_id in self.client_requests:
                    self.client_requests[request_id]["completion_time"] = time.time()
                    self.client_requests[request_id]["status_code"] = context.data.get(
                        "response_status", 200
                    )
                return
            # If we have metrics, continue to log immediately (non-streaming wrapped as streaming)

        # For non-streaming responses, log immediately
        # Get and remove request data
        request_data = self.client_requests.pop(request_id)

        # Calculate duration
        duration_ms = (time.time() - request_data["start_time"]) * 1000

        # Extract response data
        status_code = context.data.get("status_code", 200)
        body_size = context.data.get("body_size", 0)

        # Check if we have usage metrics in context metadata
        # These might be available from RequestContext metadata
        usage_metrics = {}
        if context.metadata:
            # Extract any token/cost metrics from metadata
            token_fields = [
                "tokens_input",
                "tokens_output",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost_usd",
                "model",
            ]
            for field in token_fields:
                value = context.metadata.get(field)
                if value is not None:
                    usage_metrics[field] = value

        # Merge request and response data
        log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": status_code,
            "body_size": body_size,
            "duration_ms": duration_ms,
            "error": None,
            **usage_metrics,  # Include any usage metrics found
        }

        # Format and write
        if self.client_writer:
            formatted = self.formatter.format_client(
                log_data, self.config.client_format
            )
            await self.client_writer.write(formatted)

        # Also log to structured logger
        await self._log_to_structured_logger(log_data, "client")

        # Ingest into analytics if available
        await self._maybe_ingest(log_data)

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

        # Ingest into analytics if available
        await self._maybe_ingest(log_data)

    async def _handle_provider_request(self, context: HookContext) -> None:
        """Handle PROVIDER_REQUEST_PREPARED event."""
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

    async def _handle_provider_stream_end(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_END event to capture complete streaming metrics."""
        if not self.config.provider_enabled and not self.config.client_enabled:
            return

        request_id = context.metadata.get("request_id", "unknown")

        # Extract usage metrics from the event
        usage_metrics = context.data.get("usage_metrics", {})

        # Store metrics for logging
        self._streaming_metrics[request_id] = {
            "usage_metrics": usage_metrics,
            "provider": context.provider or context.data.get("provider", "unknown"),
            "url": context.data.get("url", ""),
            "method": context.data.get("method", "POST"),
            "total_chunks": context.data.get("total_chunks", 0),
            "total_bytes": context.data.get("total_bytes", 0),
        }

        # If we have client request data for this streaming request, log it now with metrics
        if self.config.client_enabled and request_id in self.client_requests:
            request_data = self.client_requests.pop(request_id)

            # Calculate duration
            completion_time = request_data.get("completion_time", time.time())
            duration_ms = (completion_time - request_data["start_time"]) * 1000

            # Extract metrics (handle both naming conventions)
            tokens_input = usage_metrics.get(
                "input_tokens", usage_metrics.get("tokens_input", 0)
            )
            tokens_output = usage_metrics.get(
                "output_tokens", usage_metrics.get("tokens_output", 0)
            )
            cache_read_tokens = usage_metrics.get(
                "cache_read_input_tokens", usage_metrics.get("cache_read_tokens", 0)
            )
            cache_write_tokens = usage_metrics.get(
                "cache_creation_input_tokens",
                usage_metrics.get("cache_write_tokens", 0),
            )
            cost_usd = usage_metrics.get("cost_usd", 0.0)
            model = usage_metrics.get("model") or request_data.get("model", "")

            # Build complete log data
            client_log_data = {
                **request_data,
                "request_id": request_id,
                "status_code": request_data.get("status_code", 200),
                "duration_ms": duration_ms,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cost_usd": cost_usd,
                "model": model,
                "streaming": True,
                "total_chunks": context.data.get("total_chunks", 0),
                "total_bytes": context.data.get("total_bytes", 0),
                "error": None,
            }

            # Format and write client log
            if self.client_writer:
                formatted = self.formatter.format_client(
                    client_log_data, self.config.client_format
                )
                await self.client_writer.write(formatted)

            # Log to structured logger
            await self._log_to_structured_logger(client_log_data, "client")

            # Ingest into analytics with full client details (includes IP/UA)
            await self._maybe_ingest(client_log_data)

        # Extract complete metrics from usage_metrics (handle both naming conventions)
        tokens_input = usage_metrics.get(
            "input_tokens", usage_metrics.get("tokens_input", 0)
        )
        tokens_output = usage_metrics.get(
            "output_tokens", usage_metrics.get("tokens_output", 0)
        )
        cache_read_tokens = usage_metrics.get(
            "cache_read_input_tokens", usage_metrics.get("cache_read_tokens", 0)
        )
        cache_write_tokens = usage_metrics.get(
            "cache_creation_input_tokens", usage_metrics.get("cache_write_tokens", 0)
        )
        cost_usd = usage_metrics.get("cost_usd", 0.0)
        model = usage_metrics.get("model", "")

        # Get other data from context
        provider = context.provider or context.data.get("provider", "unknown")
        url = context.data.get("url", "")
        method = context.data.get("method", "POST")
        total_chunks = context.data.get("total_chunks", 0)
        total_bytes = context.data.get("total_bytes", 0)

        # Create log data for streaming complete
        log_data = {
            "timestamp": time.time(),
            "request_id": request_id,
            "provider": provider,
            "method": method,
            "url": url,
            "status_code": 200,  # Streaming completion implies success
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": cost_usd,
            "model": model,
            "total_chunks": total_chunks,
            "total_bytes": total_bytes,
            "streaming": True,
            "event_type": "streaming_complete",
        }

        # Format and write to provider log
        if self.provider_writer and self.config.provider_enabled:
            formatted = self.formatter.format_provider(log_data)
            await self.provider_writer.write(formatted)

        # Log provider streaming metrics captured (for debugging)
        logger.debug(
            "access_log_provider_stream_end_captured",
            request_id=request_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )

        # If client request details were not available earlier, we skip ingestion here
        # to avoid emitting incomplete records with missing IP/User-Agent.

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
        return any(path.startswith(excluded) for excluded in self.config.exclude_paths)

    async def _maybe_ingest(self, log_data: dict[str, Any]) -> None:
        """Ingest log data into analytics storage if service is available."""
        try:
            if self.ingest_service and hasattr(self.ingest_service, "ingest"):
                await self.ingest_service.ingest(log_data)
        except Exception as e:  # pragma: no cover - non-fatal
            logger.debug("access_log_ingest_failed", error=str(e))

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
        # Prepare structured log entry with all available fields
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

        # Add token and cost metrics (available for both client and provider logs)
        token_fields = [
            "tokens_input",
            "tokens_output",
            "cache_read_tokens",
            "cache_write_tokens",
            "cost_usd",
            "model",
        ]

        for field in token_fields:
            value = log_data.get(field)
            if value is not None:
                structured_data[field] = value

        # Add streaming-specific fields if present
        streaming_fields = ["streaming", "total_chunks", "total_bytes", "event_type"]
        for field in streaming_fields:
            value = log_data.get(field)
            if value is not None:
                structured_data[field] = value

        # Add service and endpoint info
        service_fields = ["endpoint", "service_type", "provider"]
        for field in service_fields:
            value = log_data.get(field)
            if value is not None:
                structured_data[field] = value

        # Add session context metadata if available
        session_fields = [
            "session_id",
            "session_type",
            "session_status",
            "session_age_seconds",
            "session_message_count",
            "session_pool_enabled",
            "session_idle_seconds",
            "session_error_count",
            "session_is_new",
        ]
        for field in session_fields:
            value = log_data.get(field)
            if value is not None:
                structured_data[field] = value

        # Add provider-specific URL if this is a provider log
        if log_type == "provider" and "url" not in structured_data:
            url = log_data.get("url")
            if url:
                structured_data["url"] = url

        # Remove None values to keep log clean
        structured_data = {k: v for k, v in structured_data.items() if v is not None}

        # Log with appropriate level - event is passed as first argument to logger methods
        if error:
            logger.warning("access_log", error=error, **structured_data)
        else:
            logger.info("access_log", **structured_data)

    async def _log_streaming_complete(
        self, request_id: str, context: HookContext
    ) -> None:
        """Log streaming completion with full metrics.

        This is called when REQUEST_COMPLETED fires for a streaming response,
        using the metrics we stored from PROVIDER_STREAM_END.
        """
        if request_id not in self.client_requests:
            return

        # Get stored metrics
        metrics_data = self._streaming_metrics.pop(request_id, {})
        usage_metrics = metrics_data.get("usage_metrics", {})

        # Get the original request data
        request_data = self.client_requests.pop(request_id)

        # Calculate duration
        duration_ms = (time.time() - request_data["start_time"]) * 1000

        # Extract metrics
        tokens_input = usage_metrics.get("tokens_input", 0)
        tokens_output = usage_metrics.get("tokens_output", 0)
        cache_read_tokens = usage_metrics.get("cache_read_tokens", 0)
        cache_write_tokens = usage_metrics.get("cache_write_tokens", 0)
        cost_usd = usage_metrics.get("cost_usd", 0.0)
        model = usage_metrics.get("model", "")

        # Merge request data with streaming metrics
        client_log_data = {
            **request_data,
            "request_id": request_id,
            "status_code": 200,
            "duration_ms": duration_ms,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "cost_usd": cost_usd,
            "model": model,
            "streaming": True,
            "total_chunks": metrics_data.get("total_chunks", 0),
            "total_bytes": metrics_data.get("total_bytes", 0),
            "error": None,
        }

        # Format and write client log
        if self.client_writer:
            formatted = self.formatter.format_client(
                client_log_data, self.config.client_format
            )
            await self.client_writer.write(formatted)

        # Log to structured logger for client
        await self._log_to_structured_logger(client_log_data, "client")

        logger.info(
            "access_log", **{k: v for k, v in client_log_data.items() if v is not None}
        )

    async def close(self) -> None:
        """Close writers and flush any pending data."""
        if self.client_writer:
            await self.client_writer.close()
        if self.provider_writer:
            await self.provider_writer.close()
