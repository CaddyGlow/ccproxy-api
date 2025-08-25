"""Core request tracer implementation for proxy service."""

import json
import logging
import os
from pathlib import Path

import structlog

from ccproxy.services.tracing.interfaces import RequestTracer, StreamingTracer


logger = structlog.get_logger(__name__)

# Import TRACE_LEVEL
try:
    from ccproxy.core.logging import TRACE_LEVEL
except ImportError:
    TRACE_LEVEL = 5  # Fallback


class CoreRequestTracer(RequestTracer, StreamingTracer):
    """Core proxy request tracer using TRACE level logging."""

    def __init__(
        self, verbose_api: bool = False, request_log_dir: str | None = None
    ) -> None:
        """Initialize with verbosity settings from environment.

        - Reads CCPROXY_VERBOSE_API environment variable (for backward compatibility)
        - Maps verbose_api to TRACE level logging
        - Sets up file logging directory if needed
        """
        # Keep verbose_api for backward compatibility
        self.verbose_api = (
            verbose_api or os.getenv("CCPROXY_VERBOSE_API", "").lower() == "true"
        )

        # Check if TRACE level is enabled
        current_level = (
            logger._context.get("_level", logging.INFO)
            if hasattr(logger, "_context")
            else logging.INFO
        )
        self.trace_enabled = self.verbose_api or current_level <= TRACE_LEVEL

        self.request_log_dir = request_log_dir or os.getenv("CCPROXY_REQUEST_LOG_DIR")

        if self.trace_enabled and self.request_log_dir:
            Path(self.request_log_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def redact_headers(headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers for safe logging.

        - Replaces authorization, x-api-key, cookie values with [REDACTED]
        - Preserves header names for debugging
        - Returns new dict without modifying original
        """
        sensitive_headers = {
            "authorization",
            "x-api-key",
            "api-key",
            "cookie",
            "x-auth-token",
            "x-access-token",
            "x-secret-key",
        }

        redacted = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    async def trace_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """Implementation of request tracing.

        - Logs at TRACE level with redacted headers
        - Writes to request log file with complete data (if configured)
        """
        if not self.trace_enabled:
            return

        # Log at TRACE level with redacted headers
        if hasattr(logger, "trace"):
            logger.trace(
                "api_request",
                category="http",
                request_id=request_id,
                method=method,
                url=url,
                headers=self.redact_headers(headers),
                body_size=len(body) if body else 0,
            )
        elif self.verbose_api:
            # Fallback for backward compatibility
            logger.info(
                "api_request",
                category="http",
                request_id=request_id,
                method=method,
                url=url,
                headers=self.redact_headers(headers),
                body_size=len(body) if body else 0,
            )

        # Write to file if configured
        if self.request_log_dir:
            request_file = Path(self.request_log_dir) / f"{request_id}_request.json"
            request_data = {
                "request_id": request_id,
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": body.decode("utf-8", errors="replace") if body else None,
            }
            request_file.write_text(json.dumps(request_data, indent=2))

    async def trace_response(
        self, request_id: str, status: int, headers: dict[str, str], body: bytes
    ) -> None:
        """Implementation of response tracing.

        - Logs at TRACE level
        - Truncates body preview at 1024 chars for console
        - Handles binary data gracefully
        """
        if not self.trace_enabled:
            return

        body_preview = self._get_body_preview(body)

        # Log at TRACE level
        if hasattr(logger, "trace"):
            logger.trace(
                "api_response",
                category="http",
                request_id=request_id,
                status=status,
                headers=dict(headers),
                body_preview=body_preview,
                body_size=len(body),
            )
        elif self.verbose_api:
            # Fallback for backward compatibility
            logger.info(
                "api_response",
                category="http",
                request_id=request_id,
                status=status,
                headers=dict(headers),
                body_preview=body_preview,
                body_size=len(body),
            )

        # Write to file if configured
        if self.request_log_dir:
            response_file = Path(self.request_log_dir) / f"{request_id}_response.json"
            response_data = {
                "request_id": request_id,
                "status": status,
                "headers": dict(headers),
                "body": body.decode("utf-8", errors="replace"),
            }
            response_file.write_text(json.dumps(response_data, indent=2))

    def _get_body_preview(self, body: bytes, max_length: int = 1024) -> str:
        """Extract readable preview from body bytes.

        - Decodes UTF-8 with error replacement
        - Truncates to max_length
        - Returns '<binary data>' for non-text content
        """
        try:
            text = body.decode("utf-8", errors="replace")

            # Try to parse as JSON for better formatting
            try:
                json_data = json.loads(text)
                formatted = json.dumps(json_data, indent=2)
                if len(formatted) > max_length:
                    return formatted[:max_length] + "..."
                return formatted
            except json.JSONDecodeError:
                # Not JSON, return as plain text
                if len(text) > max_length:
                    return text[:max_length] + "..."
                return text
        except UnicodeDecodeError:
            return "<binary data>"
        except Exception as e:
            logger.debug("text_formatting_unexpected_error", error=str(e))
            return "<binary data>"

    # Streaming tracer methods
    async def trace_stream_start(
        self, request_id: str, headers: dict[str, str]
    ) -> None:
        """Mark beginning of stream with initial headers."""
        if not self.verbose_api:
            return

        logger.info(
            "Starting stream response", request_id=request_id, headers=dict(headers)
        )

    async def trace_stream_chunk(
        self, request_id: str, chunk: bytes, chunk_number: int
    ) -> None:
        """Record individual stream chunk (optional, for deep debugging)."""
        # Only log if extremely verbose debugging is needed
        if os.getenv("CCPROXY_TRACE_CHUNKS", "").lower() == "true":
            logger.debug(
                "Stream chunk",
                request_id=request_id,
                chunk_number=chunk_number,
                chunk_size=len(chunk),
            )

    async def trace_stream_complete(
        self, request_id: str, total_chunks: int, total_bytes: int
    ) -> None:
        """Mark stream completion with statistics."""
        if not self.verbose_api:
            return

        logger.info(
            "Stream complete",
            request_id=request_id,
            total_chunks=total_chunks,
            total_bytes=total_bytes,
        )
