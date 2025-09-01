"""Unified request tracer implementation."""

from collections.abc import Sequence

import structlog

from ccproxy.services.tracing.interfaces import RequestTracer, StreamingTracer

from .config import RequestTracerConfig
from .formatters import JSONFormatter, RawHTTPFormatter


logger = structlog.get_logger(__name__)


class RequestTracerImpl(RequestTracer, StreamingTracer):
    """Unified request tracer with structured JSON and raw HTTP logging.

    This tracer combines:
    - Structured JSON logging for observability (from core_tracer)
    - Raw HTTP protocol logging for debugging (from raw_http_logger)
    - Streaming support for SSE/chunked responses
    """

    def __init__(self, config: RequestTracerConfig) -> None:
        """Initialize the tracer with configuration.

        Args:
            config: Unified configuration for tracing
        """
        self.config = config
        self.enabled = config.enabled

        # Initialize formatters
        self.json_formatter = JSONFormatter(config) if config.enabled else None
        self.raw_formatter = RawHTTPFormatter(config) if config.enabled else None

        # For backward compatibility
        self.verbose_api = config.verbose_api
        self.request_log_dir = (
            config.get_json_log_dir() if config.json_logs_enabled else None
        )

        if self.enabled:
            logger.info(
                "request_tracer_initialized",
                verbose_api=config.verbose_api,
                json_logs=config.json_logs_enabled,
                raw_http=config.raw_http_enabled,
                log_dir=config.log_dir,
            )

    # RequestTracer interface implementation

    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """Record request details for debugging/monitoring.

        Delegates to JSON formatter for structured logging.
        Raw HTTP logging is handled by middleware/transport.
        """
        if not self.enabled:
            return

        if self.json_formatter:
            await self.json_formatter.log_request(
                request_id=request_id,
                method=method,
                url=url,
                headers=headers,
                body=body,
            )

    async def trace_response(
        self,
        request_id: str,
        status: int,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        """Record response details.

        Delegates to JSON formatter for structured logging.
        Raw HTTP logging is handled by middleware/transport.
        """
        if not self.enabled:
            return

        if self.json_formatter:
            await self.json_formatter.log_response(
                request_id=request_id,
                status=status,
                headers=headers,
                body=body,
            )

    # StreamingTracer interface implementation

    async def trace_stream_start(
        self,
        request_id: str,
        headers: dict[str, str],
    ) -> None:
        """Mark beginning of stream with initial headers."""
        if not self.enabled:
            return

        if self.json_formatter:
            await self.json_formatter.log_stream_start(request_id)

    async def trace_stream_chunk(
        self,
        request_id: str,
        chunk: bytes,
        chunk_number: int,
    ) -> None:
        """Record individual stream chunk (optional, for deep debugging)."""
        if not self.enabled:
            return

        if self.json_formatter:
            await self.json_formatter.log_stream_chunk(request_id, chunk, chunk_number)

    async def trace_stream_complete(
        self,
        request_id: str,
        total_chunks: int,
        total_bytes: int,
    ) -> None:
        """Mark stream completion with statistics."""
        if not self.enabled:
            return

        if self.json_formatter:
            await self.json_formatter.log_stream_complete(
                request_id, None, total_chunks, total_bytes
            )

    # Raw HTTP logging methods (used by middleware/transport)

    async def log_raw_client_request(self, request_id: str, raw_data: bytes) -> None:
        """Log raw client request data."""
        if not self.enabled or not self.raw_formatter:
            return

        await self.raw_formatter.log_client_request(request_id, raw_data)

    async def log_raw_client_response(self, request_id: str, raw_data: bytes) -> None:
        """Log raw client response data."""
        if not self.enabled or not self.raw_formatter:
            return

        await self.raw_formatter.log_client_response(request_id, raw_data)

    async def log_raw_provider_request(self, request_id: str, raw_data: bytes) -> None:
        """Log raw provider request data."""
        if not self.enabled or not self.raw_formatter:
            return

        await self.raw_formatter.log_provider_request(request_id, raw_data)

    async def log_raw_provider_response(self, request_id: str, raw_data: bytes) -> None:
        """Log raw provider response data."""
        if not self.enabled or not self.raw_formatter:
            return

        await self.raw_formatter.log_provider_response(request_id, raw_data)

    # Helper methods for middleware/transport

    def should_log_raw(self) -> bool:
        """Check if raw HTTP logging is enabled."""
        return bool(
            self.enabled and self.raw_formatter and self.raw_formatter.should_log()
        )

    def should_trace_path(self, path: str) -> bool:
        """Check if a path should be traced based on include/exclude rules."""
        if not self.enabled:
            return False
        return self.config.should_trace_path(path)

    def build_raw_request(
        self,
        method: str,
        url: str,
        headers: Sequence[tuple[bytes | str, bytes | str]],
        body: bytes | None = None,
    ) -> bytes:
        """Build raw HTTP/1.1 request format."""
        if not self.raw_formatter:
            return b""
        return self.raw_formatter.build_raw_request(method, url, headers, body)

    def build_raw_response(
        self,
        status_code: int,
        headers: Sequence[tuple[bytes | str, bytes | str]],
        reason: str = "OK",
    ) -> bytes:
        """Build raw HTTP/1.1 response headers."""
        if not self.raw_formatter:
            return b""
        return self.raw_formatter.build_raw_response(status_code, headers, reason)

    @staticmethod
    def redact_headers(headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers for safe logging.

        Static method for backward compatibility.
        """
        return JSONFormatter.redact_headers(headers)
