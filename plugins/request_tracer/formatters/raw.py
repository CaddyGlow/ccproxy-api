"""Raw HTTP formatter for protocol-level logging."""

from collections.abc import Sequence
from pathlib import Path

import aiofiles

from ccproxy.core.logging import get_plugin_logger

from ..config import RequestTracerConfig


logger = get_plugin_logger()


class RawHTTPFormatter:
    """Formats and logs raw HTTP protocol data."""

    def __init__(self, config: RequestTracerConfig) -> None:
        """Initialize with configuration.

        Args:
            config: RequestTracerConfig instance
        """
        self.config = config
        self.enabled = config.raw_http_enabled
        self.log_dir = Path(config.get_raw_log_dir())
        self._log_client_request = config.log_client_request
        self._log_client_response = config.log_client_response
        self._log_provider_request = config.log_provider_request
        self._log_provider_response = config.log_provider_response
        self.max_body_size = config.max_body_size
        self.exclude_headers = [h.lower() for h in config.exclude_headers]

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

        # Track which files we've already logged to (to only log once)
        self._logged_files: set[str] = set()

    def should_log(self) -> bool:
        """Check if raw logging is enabled."""
        return bool(self.enabled)

    async def log_client_request(self, request_id: str, raw_data: bytes) -> None:
        """Log raw client request data."""
        if not self.enabled or not self._log_client_request:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        file_path = self.log_dir / f"{request_id}_client_request.http"
        file_key = f"{request_id}_client_request"

        # Only log on first write to this file
        if file_key not in self._logged_files:
            self._logged_files.add(file_key)
            logger.debug(
                "raw_http_log_started",
                request_id=request_id,
                log_type="client_request",
                file_path=str(file_path),
                category="middleware",
            )

        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    async def log_client_response(self, request_id: str, raw_data: bytes) -> None:
        """Log raw client response data."""
        if not self.enabled or not self._log_client_response:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        file_path = self.log_dir / f"{request_id}_client_response.http"
        file_key = f"{request_id}_client_response"

        # Only log on first write to this file
        if file_key not in self._logged_files:
            self._logged_files.add(file_key)
            logger.debug(
                "raw_http_log_started",
                request_id=request_id,
                log_type="client_response",
                file_path=str(file_path),
                category="middleware",
            )

        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    async def log_provider_request(self, request_id: str, raw_data: bytes) -> None:
        """Log raw provider request data."""
        if not self.enabled or not self._log_provider_request:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        file_path = self.log_dir / f"{request_id}_provider_request.http"
        file_key = f"{request_id}_provider_request"

        # Only log on first write to this file
        if file_key not in self._logged_files:
            self._logged_files.add(file_key)
            logger.debug(
                "raw_http_log_started",
                request_id=request_id,
                log_type="provider_request",
                file_path=str(file_path),
                category="middleware",
            )

        async with aiofiles.open(file_path, "ab") as f:
            await f.write(raw_data)

    async def log_provider_response(self, request_id: str, raw_data: bytes) -> None:
        """Log raw provider response data."""
        if not self.enabled or not self._log_provider_response:
            return

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"

        file_path = self.log_dir / f"{request_id}_provider_response.http"
        file_key = f"{request_id}_provider_response"

        # Only log on first write to this file
        if file_key not in self._logged_files:
            self._logged_files.add(file_key)
            logger.debug(
                "raw_http_log_started",
                request_id=request_id,
                log_type="provider_response",
                file_path=str(file_path),
                category="middleware",
            )

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

        # Add Host header if not present
        has_host = any(
            (
                h[0].lower() == b"host"
                if isinstance(h[0], bytes)
                else h[0].lower() == "host"
            )
            for h in headers
        )
        if not has_host and parsed.netloc:
            lines.append(f"Host: {parsed.netloc}")

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
