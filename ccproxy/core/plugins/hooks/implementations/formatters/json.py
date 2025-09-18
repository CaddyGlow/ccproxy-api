"""JSON formatter for structured request/response logging."""

import base64
import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import get_merged_contextvars

from ccproxy.core.plugins.hooks.types import HookHeaders


try:
    from ccproxy.core.logging import TRACE_LEVEL
except ImportError:
    TRACE_LEVEL = 5  # Fallback

logger = structlog.get_logger(__name__)


class JSONFormatter:
    """Formats requests/responses as structured JSON for observability."""

    def __init__(
        self,
        log_dir: str = "/tmp/ccproxy/traces",
        verbose_api: bool = True,
        json_logs_enabled: bool = True,
        redact_sensitive: bool = True,
        truncate_body_preview: int = 1024,
    ) -> None:
        """Initialize with configuration.

        Args:
            log_dir: Directory for log files
            verbose_api: Enable verbose API logging
            json_logs_enabled: Enable JSON file logging
            redact_sensitive: Redact sensitive headers
            truncate_body_preview: Max body preview size
        """
        self.log_dir = log_dir
        self.verbose_api = verbose_api
        self.json_logs_enabled = json_logs_enabled
        self.redact_sensitive = redact_sensitive
        self.truncate_body_preview = truncate_body_preview

        # Check if TRACE level is enabled
        current_level = (
            logger._context.get("_level", logging.INFO)
            if hasattr(logger, "_context")
            else logging.INFO
        )
        self.trace_enabled = self.verbose_api or current_level <= TRACE_LEVEL

        # Setup log directory if file logging is enabled
        self.request_log_dir = None
        if self.json_logs_enabled:
            self.request_log_dir = Path(log_dir)
            self.request_log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, config: Any) -> "JSONFormatter":
        """Create JSONFormatter from a RequestTracerConfig.

        Args:
            config: RequestTracerConfig instance

        Returns:
            JSONFormatter instance
        """
        return cls(
            log_dir=config.get_json_log_dir(),
            verbose_api=config.verbose_api,
            json_logs_enabled=config.json_logs_enabled,
            redact_sensitive=config.redact_sensitive,
            truncate_body_preview=config.truncate_body_preview,
        )

    def _current_cmd_id(self) -> str | None:
        """Return current cmd_id from structlog contextvars or env."""
        try:
            ctx = get_merged_contextvars(logger) or {}
            cmd_id = ctx.get("cmd_id")
        except Exception:
            cmd_id = None

        return str(cmd_id) if cmd_id else None

    def _compose_file_id(self, request_id: str | None) -> str:
        """Build filename ID using cmd_id and request_id per rules.

        - If both cmd_id and request_id exist: "{cmd_id}_{request_id}"
        - If only request_id exists: request_id
        - If only cmd_id exists: cmd_id
        - If neither exists: generate a UUID4
        """
        try:
            ctx = get_merged_contextvars(logger) or {}
            cmd_id = ctx.get("cmd_id")
        except Exception:
            cmd_id = None

        if cmd_id and request_id:
            return f"{cmd_id}_{request_id}"
        if request_id:
            return request_id
        if cmd_id:
            return str(cmd_id)
        return str(uuid.uuid4())

    def _compose_file_id_with_timestamp(self, request_id: str | None) -> str:
        """Build filename ID with timestamp suffix for better organization.

        Format: {base_id}_{timestamp}_{sequence}
        Where timestamp is in format: YYYYMMDD_HHMMSS_microseconds
        And sequence is a counter to prevent collisions
        """
        base_id = self._compose_file_id(request_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Add a high-resolution timestamp with nanoseconds for uniqueness
        nanos = time.time_ns() % 1000000  # Get nanosecond portion
        return f"{base_id}_{timestamp}_{nanos:06d}"

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
            "x-secret-key",
        }

        redacted = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = value
        return redacted

    async def log_request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: HookHeaders | dict[str, str],
        body: bytes | None,
        request_type: str = "provider",  # "client" or "provider"
        context: Any = None,  # RequestContext
        hook_type: str | None = None,  # Hook type for filename (e.g., "tracer", "http")
    ) -> None:
        """Log structured request data.

        - Logs at TRACE level with redacted headers
        - Writes to request log file with complete data (if configured)
        """
        if not self.trace_enabled:
            return

        # Normalize headers (preserve order/case if dict-like)
        headers_dict = (
            headers.to_dict() if hasattr(headers, "to_dict") else dict(headers)
        )

        # Log at TRACE level with redacted headers
        log_headers = (
            self.redact_headers(headers_dict) if self.redact_sensitive else headers_dict
        )

        if hasattr(logger, "trace"):
            logger.trace(
                "api_request",
                category="http",
                request_id=request_id,
                method=method,
                url=url,
                headers=log_headers,
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
                headers=log_headers,
                body_size=len(body) if body else 0,
            )

        # Write to file if configured
        if self.request_log_dir and self.json_logs_enabled:
            # Build file suffix with hook type
            base_suffix = (
                f"{request_type}_request" if request_type != "provider" else "request"
            )
            if hook_type:
                file_suffix = f"{base_suffix}_{hook_type}"
            else:
                file_suffix = base_suffix

            base_id = self._compose_file_id_with_timestamp(request_id)
            request_file = self.request_log_dir / f"{base_id}_{file_suffix}.json"

            # Handle body content - could be bytes, dict/list (from JSON), or string
            body_content = None
            if body is not None:
                if isinstance(body, dict | list):
                    # Already parsed JSON object from hook context
                    body_content = body
                elif isinstance(body, bytes):
                    # Raw bytes - try to parse as JSON first, then string, then base64
                    try:
                        # First try to decode as UTF-8 string
                        body_str = body.decode("utf-8")
                        # Then try to parse as JSON
                        body_content = json.loads(body_str)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Not JSON, try plain string
                        try:
                            body_content = body.decode("utf-8", errors="replace")
                        except Exception:
                            # Last resort: encode as base64
                            body_content = {
                                "_type": "base64",
                                "data": base64.b64encode(body).decode("ascii"),
                            }
                elif isinstance(body, str):
                    # String body - try to parse as JSON, otherwise keep as string
                    try:
                        body_content = json.loads(body)
                    except json.JSONDecodeError:
                        body_content = body
                else:
                    # Other type - convert to string
                    body_content = str(body)

            request_data = {
                "request_id": request_id,
                "method": method,
                "url": url,
                "headers": headers_dict,  # Full headers in file
                "body": body_content,
                "type": request_type,
            }

            # Add cmd_id for CLI correlation if present
            cmd_id = self._current_cmd_id()
            if cmd_id:
                request_data["cmd_id"] = cmd_id

            # Add context data if available
            if context and hasattr(context, "to_dict"):
                try:
                    context_data = context.to_dict()
                    if context_data:
                        request_data["context"] = context_data
                except Exception as e:
                    logger.debug(
                        "context_serialization_error",
                        error=str(e),
                        request_id=request_id,
                    )

            request_file.write_text(json.dumps(request_data, indent=2, default=str))

    async def log_response(
        self,
        request_id: str,
        status: int,
        headers: HookHeaders | dict[str, str],
        body: bytes,
        response_type: str = "provider",  # "client" or "provider"
        context: Any = None,  # RequestContext
        hook_type: str | None = None,  # Hook type for filename (e.g., "tracer", "http")
    ) -> None:
        """Log structured response data.

        - Logs at TRACE level
        - Truncates body preview for console
        - Handles binary data gracefully
        """
        if not self.trace_enabled:
            return

        body_preview = self._get_body_preview(body)

        # Normalize headers (preserve order/case if dict-like)
        headers_dict = (
            headers.to_dict() if hasattr(headers, "to_dict") else dict(headers)
        )

        # Log at TRACE level
        if hasattr(logger, "trace"):
            logger.trace(
                "api_response",
                category="http",
                request_id=request_id,
                status=status,
                headers=headers_dict,
                body_preview=body_preview,
                body_size=len(body),
            )
        else:
            # Fallback for backward compatibility
            logger.info(
                "api_response",
                category="http",
                request_id=request_id,
                status=status,
                headers=headers_dict,
                body_preview=body_preview,
                body_size=len(body),
            )

        # Write to file if configured
        if self.request_log_dir and self.json_logs_enabled:
            # Build file suffix with hook type
            base_suffix = (
                f"{response_type}_response"
                if response_type != "provider"
                else "response"
            )
            if hook_type:
                file_suffix = f"{base_suffix}_{hook_type}"
            else:
                file_suffix = base_suffix
            logger.debug(
                "Writing response JSON file",
                request_id=request_id,
                status=status,
                response_type=response_type,
                file_suffix=file_suffix,
                body_type=type(body).__name__,
                body_size=len(body) if body else 0,
                body_preview=body[:100] if body else None,
            )
            base_id = self._compose_file_id_with_timestamp(request_id)
            response_file = self.request_log_dir / f"{base_id}_{file_suffix}.json"

            # Try to parse body as JSON first, then string, then base64
            body_content: str | dict[str, Any] = ""
            if body:
                try:
                    # First try to decode as UTF-8 string
                    body_str = body.decode("utf-8")
                    # Then try to parse as JSON
                    body_content = json.loads(body_str)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Not JSON, try plain string
                    try:
                        body_content = body.decode("utf-8", errors="replace")
                    except Exception:
                        # Last resort: encode as base64
                        import base64

                        body_content = {
                            "_type": "base64",
                            "data": base64.b64encode(body).decode("ascii"),
                        }

            response_data = {
                "request_id": request_id,
                "status": status,
                "headers": headers_dict,
                "body": body_content,
                "type": response_type,
            }

            # Add cmd_id for CLI correlation if present
            cmd_id = self._current_cmd_id()
            if cmd_id:
                response_data["cmd_id"] = cmd_id

            # Add context data if available (including cost/metrics)
            if context and hasattr(context, "to_dict"):
                try:
                    context_data = context.to_dict()
                    if context_data:
                        response_data["context"] = context_data
                except Exception as e:
                    logger.debug(
                        "context_serialization_error",
                        error=str(e),
                        request_id=request_id,
                    )

            response_file.write_text(json.dumps(response_data, indent=2, default=str))

    def _get_body_preview(self, body: bytes) -> str:
        """Extract readable preview from body bytes.

        - Decodes UTF-8 with error replacement
        - Truncates to max_length
        - Returns '<binary data>' for non-text content
        """
        max_length = self.truncate_body_preview

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

    # Streaming methods
    async def log_stream_chunk(
        self, request_id: str, chunk: bytes, chunk_number: int
    ) -> None:
        """Record individual stream chunk (optional, for deep debugging)."""
        logger.debug(
            "stream_chunk",
            category="streaming",
            request_id=request_id,
            chunk_number=chunk_number,
            chunk_size=len(chunk),
        )

    async def log_error(
        self,
        request_id: str,
        error: Exception | None,
        duration: float | None = None,
        provider: str | None = None,
    ) -> None:
        """Log error information."""
        if not self.verbose_api:
            return

        error_data: dict[str, Any] = {
            "request_id": request_id,
            "error": str(error) if error else "unknown",
            "category": "error",
        }

        if duration is not None:
            error_data["duration"] = duration
        if provider:
            error_data["provider"] = provider

        logger.error("request_error", **error_data)

    # Legacy compatibility methods
    async def log_provider_request(
        self,
        request_id: str,
        provider: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """Log provider request."""
        await self.log_request(
            request_id=request_id,
            method=method,
            url=url,
            headers=headers,
            body=body,
            request_type="provider",
        )

    async def log_provider_response(
        self,
        request_id: str,
        provider: str,
        status_code: int,
        headers: dict[str, str],
        body: bytes | None,
    ) -> None:
        """Log provider response."""
        await self.log_response(
            request_id=request_id,
            status=status_code,
            headers=headers,
            body=body or b"",
            response_type="provider",
        )

    async def log_stream_start(
        self,
        request_id: str,
        provider: str | None = None,
    ) -> None:
        """Log stream start."""
        if not self.verbose_api:
            return

        log_data: dict[str, Any] = {
            "request_id": request_id,
            "category": "streaming",
        }
        if provider:
            log_data["provider"] = provider

        logger.info("stream_start", **log_data)

    async def log_stream_complete(
        self,
        request_id: str,
        provider: str | None = None,
        total_chunks: int | None = None,
        total_bytes: int | None = None,
        usage_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log stream completion with metrics."""
        if not self.verbose_api:
            return

        log_data: dict[str, Any] = {
            "request_id": request_id,
            "category": "streaming",
        }
        if provider:
            log_data["provider"] = provider
        if total_chunks is not None:
            log_data["total_chunks"] = total_chunks
        if total_bytes is not None:
            log_data["total_bytes"] = total_bytes
        if usage_metrics:
            log_data["usage_metrics"] = usage_metrics

        logger.info("stream_complete", **log_data)
