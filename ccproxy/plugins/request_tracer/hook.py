"""Hook-based request tracer implementation for REQUEST_* events only."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent

from .config import RequestTracerConfig


logger = get_plugin_logger(__name__)


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

        # Storage for streaming chunks per request
        self._streaming_chunks: dict[str, list[bytes]] = {}
        self._streaming_metadata: dict[str, dict[str, Any]] = {}

        logger.debug(
            "request_tracer_hook_initialized",
            enabled=self.config.enabled,
        )

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events for request tracing.

        Args:
            context: Hook context with event data
        """
        # Debug logging for CLI hook calls
        # logger.trace(
        #     "request_tracer_hook_called",
        #     hook_event=context.event.value if context.event else "unknown",
        #     enabled=self.config.enabled,
        #     data_keys=list(context.data.keys()) if context.data else [],
        # )

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

        logger.trace(
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

        logger.trace(
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

        logger.trace(
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

        logger.trace(
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

        logger.trace(
            "provider_response_received",
            request_id=request_id,
            provider=provider,
            status_code=status_code,
            is_streaming=is_streaming,
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
        request_id = context.data.get("request_id") or context.metadata.get(
            "request_id", "unknown"
        )
        provider = context.provider or "unknown"

        logger.debug(
            "stream_start_handler_called",
            chunk_logging_enabled=self.config.log_streaming_chunks,
            json_logs_enabled=self.config.json_logs_enabled,
            request_id=request_id,
            provider=provider,
        )

        if self.config.json_logs_enabled:
            # Initialize chunk collection for this request when JSON logs are enabled
            self._streaming_chunks[request_id] = []
            self._streaming_metadata[request_id] = {
                "provider": provider,
                "start_time": datetime.now(),
                "url": context.data.get("url", ""),
                "method": context.data.get("method", "UNKNOWN"),
                "buffered_mode": context.data.get("buffered_mode", False),
                "upstream_stream_text": None,
            }

        logger.debug(
            "stream_started",
            request_id=request_id,
            provider=provider,
        )

    async def _handle_stream_chunk(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_CHUNK event."""
        request_id = context.data.get("request_id", "unknown")
        chunk = context.data.get("chunk")

        if (
            chunk
            and self.config.json_logs_enabled
            and request_id in self._streaming_chunks
        ):
            # Collect the chunk
            self._streaming_chunks[request_id].append(chunk)

        if not self.config.log_streaming_chunks:
            return

        # Optional: Log chunk info for debugging
        chunk_number = context.data.get("chunk_number", 0)
        chunk_size = context.data.get("chunk_size", len(chunk) if chunk else 0)

        logger.trace(
            "stream_chunk_collected",
            request_id=request_id,
            chunk_number=chunk_number,
            chunk_size=chunk_size,
        )

    async def _handle_stream_end(self, context: HookContext) -> None:
        """Handle PROVIDER_STREAM_END event."""
        request_id = context.data.get("request_id", "unknown")
        provider = context.provider or "unknown"
        total_chunks = context.data.get("total_chunks", 0)
        total_bytes = context.data.get("total_bytes", 0)
        usage_metrics = context.data.get("usage_metrics", {})

        # Write collected chunks to response file
        if self.config.json_logs_enabled:
            if (
                request_id in self._streaming_chunks
                and self._streaming_chunks[request_id]
            ):
                metadata = self._streaming_metadata.get(request_id, {})
                chunks = self._streaming_chunks[request_id]

                # Add end time and metrics to metadata
                metadata.update(
                    {
                        "end_time": datetime.now(),
                        "total_chunks": total_chunks,
                        "total_bytes": total_bytes,
                        "usage_metrics": usage_metrics,
                        "upstream_stream_text": context.data.get(
                            "upstream_stream_text"
                        ),
                    }
                )

                # Write response file
                await self._write_streaming_response_file(request_id, chunks, metadata)

            # Clean up memory regardless of whether we had chunks
            self._streaming_chunks.pop(request_id, None)
            self._streaming_metadata.pop(request_id, None)

        logger.trace(
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

    def _generate_stream_response_file_path(
        self, request_id: str, provider: str
    ) -> Path:
        """Generate file path for streaming response file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{request_id}_{timestamp}_{provider}_streaming_response.json"
        return Path(self.config.get_json_log_dir()) / filename

    async def _write_streaming_response_file(
        self, request_id: str, chunks: list[bytes], metadata: dict[str, Any]
    ) -> None:
        """Write collected streaming chunks to a response file."""
        try:
            # Combine all chunks
            combined_data = b"".join(chunks)

            # Try to decode as text for JSON parsing
            try:
                response_text = combined_data.decode("utf-8", errors="replace")
            except Exception:
                response_text = str(combined_data)

            # Build response data
            upstream_stream_text = metadata.get("upstream_stream_text")
            response_data = {
                "request_id": request_id,
                "provider": metadata.get("provider", "unknown"),
                "method": metadata.get("method", "UNKNOWN"),
                "url": metadata.get("url", ""),
                "start_time": metadata.get("start_time", datetime.now()).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_chunks": len(chunks),
                "total_bytes": len(combined_data),
                "buffered_mode": metadata.get("buffered_mode", False),
                "usage_metrics": metadata.get("usage_metrics"),
                # "response_text": response_text[: self.config.truncate_body_preview]
                # if len(response_text) > self.config.truncate_body_preview
                # else response_text,
                # "response_truncated": len(response_text)
                # > self.config.truncate_body_preview,
                "response_text": response_text,
            }

            if upstream_stream_text is not None:
                response_data["upstream_stream_text"] = upstream_stream_text

            # Generate file path
            file_path = self._generate_stream_response_file_path(
                request_id, metadata.get("provider", "unknown")
            )

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON file
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(response_data, indent=2, ensure_ascii=False))

            logger.debug(
                "streaming_response_file_written",
                request_id=request_id,
                file_path=str(file_path),
                total_chunks=len(chunks),
                total_bytes=len(combined_data),
            )

        except Exception as e:
            logger.error(
                "streaming_response_file_write_failed",
                request_id=request_id,
                error=str(e),
                exc_info=e,
            )
