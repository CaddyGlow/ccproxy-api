"""Hook-based request tracer implementation - simplified for REQUEST_* events only."""

import structlog

from ccproxy.hooks import Hook
from ccproxy.hooks.base import HookContext
from ccproxy.hooks.events import HookEvent

from .config import RequestTracerConfig


logger = structlog.get_logger(__name__)


class RequestTracerHook(Hook):
    """Simplified hook-based request tracer implementation.

    This hook only handles REQUEST_* events since HTTP_* events are now
    handled by the core HTTPTracerHook. This eliminates duplication and
    follows the single responsibility principle.

    The plugin now focuses purely on request lifecycle logging without
    attempting to capture HTTP request/response bodies.
    """

    name = "request_tracer"
    events = [
        HookEvent.REQUEST_STARTED,
        HookEvent.REQUEST_COMPLETED,
        HookEvent.REQUEST_FAILED,
        # Legacy provider events for compatibility
        HookEvent.PROVIDER_REQUEST_SENT,
        HookEvent.PROVIDER_RESPONSE_RECEIVED,
        HookEvent.PROVIDER_ERROR,
        HookEvent.PROVIDER_STREAM_START,
        HookEvent.PROVIDER_STREAM_CHUNK,
        HookEvent.PROVIDER_STREAM_END,
    ]
    priority = 300  # HookLayer.ENRICHMENT - Capture/enrich request context early

    def __init__(
        self,
        config: RequestTracerConfig | None = None,
    ) -> None:
        """Initialize the request tracer hook.

        Args:
            config: Request tracer configuration
        """
        self.config = config or RequestTracerConfig()

        logger.info(
            "request_tracer_hook_initialized",
            enabled=self.config.enabled,
            note="Simplified - HTTP tracing handled by core HTTPTracerHook",
        )

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events for request tracing.

        Args:
            context: Hook context with event data
        """
        # Debug logging for CLI hook calls
        logger.debug(
            "request_tracer_hook_called",
            hook_event=context.event.value if context.event else "unknown",
            enabled=self.config.enabled,
            data_keys=list(context.data.keys()) if context.data else [],
        )

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
            HookEvent.PROVIDER_STREAM_START: self._handle_stream_start,
            HookEvent.PROVIDER_STREAM_CHUNK: self._handle_stream_chunk,
            HookEvent.PROVIDER_STREAM_END: self._handle_stream_end,
        }

        handler = handlers.get(context.event)
        if handler:
            try:
                await handler(context)
            except Exception as e:
                logger.error(
                    "request_tracer_hook_error",
                    hook_event=context.event.value if context.event else "unknown",
                    error=str(e),
                    exc_info=e,
                )

    async def _handle_request_start(self, context: HookContext) -> None:
        """Handle REQUEST_STARTED event."""
        if not self.config.log_client_request:
            return

        # Extract request data from context
        request_id = context.data.get("request_id", "unknown")
        method = context.data.get("method", "UNKNOWN")
        url = context.data.get("url", "")
        path = context.data.get("path", url)  # Use direct path if available

        # Check path filters
        if self._should_exclude_path(path):
            return

        logger.debug(
            "request_started",
            request_id=request_id,
            method=method,
            url=url,
            note="Request body logged by core HTTPTracerHook",
        )

    async def _handle_request_complete(self, context: HookContext) -> None:
        """Handle REQUEST_COMPLETED event."""
        if not self.config.log_client_response:
            return

        request_id = context.data.get("request_id", "unknown")
        status_code = context.data.get("status_code", 200)
        duration_ms = context.data.get("duration_ms", 0)

        # Check path filters
        url = context.data.get("url", "")
        path = self._extract_path(url)
        if self._should_exclude_path(path):
            return

        logger.debug(
            "request_completed",
            request_id=request_id,
            status_code=status_code,
            duration_ms=duration_ms,
            note="Response body logged by core HTTPTracerHook",
        )

    async def _handle_request_failed(self, context: HookContext) -> None:
        """Handle REQUEST_FAILED event."""
        request_id = context.data.get("request_id", "unknown")
        error = context.error
        duration = context.data.get("duration", 0)

        logger.error(
            "request_failed",
            request_id=request_id,
            error=str(error) if error else "unknown",
            duration=duration,
        )

    async def _handle_provider_request(self, context: HookContext) -> None:
        """Handle PROVIDER_REQUEST_SENT event."""
        if not self.config.log_provider_request:
            return

        request_id = context.metadata.get("request_id", "unknown")
        url = context.data.get("url", "")
        method = context.data.get("method", "UNKNOWN")
        provider = context.provider or "unknown"

        logger.debug(
            "provider_request_sent",
            request_id=request_id,
            provider=provider,
            method=method,
            url=url,
            note="Request body logged by core HTTPTracerHook",
        )

    async def _handle_provider_response(self, context: HookContext) -> None:
        """Handle PROVIDER_RESPONSE_RECEIVED event."""
        if not self.config.log_provider_response:
            return

        request_id = context.metadata.get("request_id", "unknown")
        status_code = context.data.get("status_code", 200)
        provider = context.provider or "unknown"
        is_streaming = context.data.get("is_streaming", False)

        logger.debug(
            "provider_response_received",
            request_id=request_id,
            provider=provider,
            status_code=status_code,
            is_streaming=is_streaming,
            note="Response body logged by core HTTPTracerHook",
        )

    async def _handle_provider_error(self, context: HookContext) -> None:
        """Handle PROVIDER_ERROR event."""
        request_id = context.metadata.get("request_id", "unknown")
        provider = context.provider or "unknown"
        error = context.error

        logger.error(
            "provider_error",
            request_id=request_id,
            provider=provider,
            error=str(error) if error else "unknown",
        )

    async def _handle_stream_start(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_START event."""
        if not self.config.log_streaming_chunks:
            return

        request_id = context.data.get("request_id", "unknown")
        provider = context.provider or "unknown"

        logger.debug(
            "stream_started",
            request_id=request_id,
            provider=provider,
        )

    async def _handle_stream_chunk(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_CHUNK event."""
        if not self.config.log_streaming_chunks:
            return

        # Note: We might want to skip individual chunks for performance
        # This is just a placeholder for potential chunk processing
        pass

    async def _handle_stream_end(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_END event."""
        if not self.config.log_streaming_chunks:
            return

        request_id = context.data.get("request_id", "unknown")
        provider = context.provider or "unknown"
        total_chunks = context.data.get("total_chunks", 0)
        total_bytes = context.data.get("total_bytes", 0)
        usage_metrics = context.data.get("usage_metrics", {})

        logger.debug(
            "stream_ended",
            request_id=request_id,
            provider=provider,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
            usage_metrics=usage_metrics,
        )

    def _extract_path(self, url: str) -> str:
        """Extract path from URL."""
        if "://" in url:
            # Full URL
            parts = url.split("/", 3)
            return "/" + parts[3] if len(parts) > 3 else "/"
        return url

    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from logging."""
        # Check include paths first (if specified)
        if self.config.include_paths:
            return not any(path.startswith(p) for p in self.config.include_paths)

        # Check exclude paths
        if self.config.exclude_paths:
            return any(path.startswith(p) for p in self.config.exclude_paths)

        return False
