"""Raw HTTP formatter for protocol-level logging."""

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiofiles
import structlog
from structlog.contextvars import get_merged_contextvars

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class RawHTTPFormatter:
    """Formats and logs raw HTTP protocol data."""

    def __init__(
        self,
        log_dir: str = "/tmp/ccproxy/traces",
        enabled: bool = True,
        log_client_request: bool = True,
        log_client_response: bool = True,
        log_provider_request: bool = True,
        log_provider_response: bool = True,
        max_body_size: int = 10485760,  # 10MB
        exclude_headers: list[str] | None = None,
    ) -> None:
        """Initialize with configuration.

        Args:
            log_dir: Directory for raw HTTP log files
            enabled: Enable raw HTTP logging
            log_client_request: Log client requests
            log_client_response: Log client responses
            log_provider_request: Log provider requests
            log_provider_response: Log provider responses
            max_body_size: Maximum body size to log
            exclude_headers: Headers to redact
        """
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self._log_client_request = log_client_request
        self._log_client_response = log_client_response
        self._log_provider_request = log_provider_request
        self._log_provider_response = log_provider_response
        self.max_body_size = max_body_size
        self.exclude_headers = [
            h.lower()
            for h in (
                exclude_headers
                or ["authorization", "x-api-key", "cookie", "x-auth-token"]
            )
        ]

        if self.enabled:
            # Create log directory if it doesn't exist
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error(
                    "failed_to_create_raw_log_directory",
                    log_dir=str(self.log_dir),
                    error=str(e),
                    exc_info=e,
                )
                # Disable logging if we can't create the directory
                self.enabled = False

        # Track which files we've already created (for logging purposes only)
        self._created_files: set[str] = set()

    @classmethod
    def from_config(cls, config: Any) -> "RawHTTPFormatter":
        """Create RawHTTPFormatter from a RequestTracerConfig.

        Args:
            config: RequestTracerConfig instance

        Returns:
            RawHTTPFormatter instance
        """
        return cls(
            log_dir=config.get_raw_log_dir(),
            enabled=config.raw_http_enabled,
            log_client_request=config.log_client_request,
            log_client_response=config.log_client_response,
            log_provider_request=config.log_provider_request,
            log_provider_response=config.log_provider_response,
            max_body_size=config.max_body_size,
            exclude_headers=config.exclude_headers,
        )

    def _compose_file_id(self, request_id: str | None) -> str:
        """Build filename ID using cmd_id and request_id per rules.

        - If both cmd_id and request_id exist: "{cmd_id}_{request_id}"
        - If only request_id exists: request_id
        - If only cmd_id exists: cmd_id
        - If neither exists: generate a UUID4
        """
        try:
            # structlog's typing expects a BindableLogger; use a fresh one
            ctx = get_merged_contextvars(structlog.get_logger()) or {}
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
        import time
        from datetime import datetime

        base_id = self._compose_file_id(request_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Add a high-resolution timestamp with nanoseconds for uniqueness
        nanos = time.time_ns() % 1000000  # Get nanosecond portion
        return f"{base_id}_{timestamp}_{nanos:06d}"

    def should_log(self) -> bool:
        """Check if raw logging is enabled."""
        return bool(self.enabled)

    async def log_client_request(
        self, request_id: str, raw_data: bytes, hook_type: str | None = None
    ) -> None:
        """Log raw client request data."""
        if not self.enabled or not self._log_client_request:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        base_id = self._compose_file_id_with_timestamp(request_id)
        base_suffix = "client_request"
        if hook_type:
            file_suffix = f"{base_suffix}_{hook_type}"
        else:
            file_suffix = base_suffix
        file_path = self.log_dir / f"{base_id}_{file_suffix}.http"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "raw_http_log_created",
                request_id=request_id,
                log_type="client_request",
                file_path=str(file_path),
                category="raw_formatter",
            )

        # Write data to file (append mode for multiple chunks)
        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    async def log_client_response(
        self, request_id: str, raw_data: bytes, hook_type: str | None = None
    ) -> None:
        """Log raw client response data."""
        if not self.enabled or not self._log_client_response:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        base_id = self._compose_file_id_with_timestamp(request_id)
        base_suffix = "client_response"
        if hook_type:
            file_suffix = f"{base_suffix}_{hook_type}"
        else:
            file_suffix = base_suffix
        file_path = self.log_dir / f"{base_id}_{file_suffix}.http"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "raw_http_log_created",
                request_id=request_id,
                log_type="client_response",
                file_path=str(file_path),
                category="raw_formatter",
                length=len(raw_data),
            )

        # Write data to file (append mode for multiple chunks)
        logger.debug("open_file_", length=len(raw_data), file_path=str(file_path))

        # Note: Async file write is only creating the file
        # and not writing data.
        # It seem to block the event loop and make the following hook to not execute
        # for example the request.completed
        # sync write seem to solve the issue
        # with Path(file_path).open("ab") as sync_f:
        #     sync_f.write(raw_data)
        async with aiofiles.open(file_path, "wb") as f:
            logger.debug("writing_raw_data", length=len(raw_data))
            await f.write(raw_data)

        logger.debug("finish_to_write", length=len(raw_data), file_path=str(file_path))

    async def log_provider_request(
        self, request_id: str, raw_data: bytes, hook_type: str | None = None
    ) -> None:
        """Log raw provider request data."""
        if not self.enabled or not self._log_provider_request:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        base_id = self._compose_file_id_with_timestamp(request_id)
        base_suffix = "provider_request"
        if hook_type:
            file_suffix = f"{base_suffix}_{hook_type}"
        else:
            file_suffix = base_suffix
        file_path = self.log_dir / f"{base_id}_{file_suffix}.http"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "raw_http_log_created",
                request_id=request_id,
                log_type="provider_request",
                file_path=str(file_path),
                category="raw_formatter",
            )

        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    async def log_provider_response(
        self, request_id: str, raw_data: bytes, hook_type: str | None = None
    ) -> None:
        """Log raw provider response data."""
        if not self.enabled or not self._log_provider_response:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        base_id = self._compose_file_id_with_timestamp(request_id)
        base_suffix = "provider_response"
        if hook_type:
            file_suffix = f"{base_suffix}_{hook_type}"
        else:
            file_suffix = base_suffix
        file_path = self.log_dir / f"{base_id}_{file_suffix}.http"

        # Log file creation (only once per unique file path)
        if str(file_path) not in self._created_files:
            self._created_files.add(str(file_path))
            logger.debug(
                "raw_http_log_created",
                request_id=request_id,
                log_type="provider_response",
                file_path=str(file_path),
                category="raw_formatter",
            )

        # Write data to file (append mode for multiple chunks)
        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    def build_raw_request(
        self,
        method: str,
        url: str,
        headers: Sequence[tuple[bytes | str, bytes | str]],
        body: bytes | None = None,
    ) -> bytes:
        """Build raw HTTP/1.1 request format."""
        # Parse URL to get path
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path = parsed.path or "/"
        if parsed.query:
            path += f"?{parsed.query}"

        # Build request line
        lines = [f"{method} {path} HTTP/1.1"]

        # # Add Host header if not present
        # has_host = any(
        #     (
        #         h[0].lower() == b"host"
        #         if isinstance(h[0], bytes)
        #         else h[0].lower() == "host"
        #     )
        #     for h in headers
        # )
        # if not has_host and parsed.netloc:
        #     lines.append(f"Host: {parsed.netloc}")
        #
        # Add headers with optional redaction
        for name, value in headers:
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="ignore")

            # Check if header should be redacted
            if name.lower() in self.exclude_headers:
                lines.append(f"{name}: [REDACTED]")
            else:
                lines.append(f"{name}: {value}")

        # Build raw request
        raw = "\r\n".join(lines).encode("utf-8")
        raw += b"\r\n\r\n"

        # Add body if present
        if body:
            raw += body

        return raw

    def build_raw_response(
        self,
        status_code: int,
        headers: Sequence[tuple[bytes | str, bytes | str]],
        reason: str = "OK",
    ) -> bytes:
        """Build raw HTTP/1.1 response headers."""
        # Build status line
        lines = [f"HTTP/1.1 {status_code} {reason}"]

        # Add headers with optional redaction
        for name, value in headers:
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="ignore")

            # Check if header should be redacted
            if name.lower() in self.exclude_headers:
                lines.append(f"{name}: [REDACTED]")
            else:
                lines.append(f"{name}: {value}")

        # Build raw response headers
        raw = "\r\n".join(lines).encode("utf-8")
        raw += b"\r\n\r\n"

        return raw
