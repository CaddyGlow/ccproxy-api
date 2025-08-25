"""Raw HTTP logger for direct transport-level logging."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import aiofiles

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class RawHTTPLogger:
    """Direct logger for raw HTTP data without buffering."""

    def __init__(self, config: Any = None) -> None:
        """Initialize logger with configuration.

        Args:
            config: RawHTTPLoggerConfig instance
        """
        self.config = config
        if config:
            self.enabled = config.enabled
            self.log_dir = Path(config.log_dir)
            self._log_client_request = config.log_client_request
            self._log_client_response = config.log_client_response
            self._log_provider_request = config.log_provider_request
            self._log_provider_response = config.log_provider_response
            self.max_body_size = config.max_body_size
        else:
            # Fallback for backward compatibility
            import os

            self.enabled = os.getenv("CCPROXY_LOG_RAW_HTTP", "").lower() == "true"
            self.log_dir = Path(os.getenv("CCPROXY_RAW_LOG_DIR", "/tmp/ccproxy/raw"))
            self._log_client_request = True
            self._log_client_response = True
            self._log_provider_request = True
            self._log_provider_response = True
            self.max_body_size = 10485760

        if self.enabled:
            # Create log directory if it doesn't exist
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Track which files we've already logged to (to only log once)
        self._logged_files: set[str] = set()

    def should_log(self) -> bool:
        """Check if logging is enabled."""
        return bool(self.enabled)

    async def log_client_request(self, request_id: str, raw_data: bytes) -> None:
        """Log raw client request data."""
        if not self.enabled or not self._log_client_request:
            return

        original_size = len(raw_data)
        truncated = False

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"
            truncated = True

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

        original_size = len(raw_data)
        truncated = False

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"
            truncated = True

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

        original_size = len(raw_data)
        truncated = False

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"
            truncated = True

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

        original_size = len(raw_data)
        truncated = False

        # Truncate if too large
        if len(raw_data) > self.max_body_size:
            raw_data = raw_data[: self.max_body_size] + b"\n[TRUNCATED]"
            truncated = True

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
        has_host = any(h[0].lower() == b"host" for h in headers)
        if not has_host and parsed.netloc:
            lines.append(f"Host: {parsed.netloc}")

        # Add headers
        for name, value in headers:
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="ignore")
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

        # Add headers
        for name, value in headers:
            if isinstance(name, bytes):
                name = name.decode("ascii", errors="ignore")
            if isinstance(value, bytes):
                value = value.decode("ascii", errors="ignore")
            lines.append(f"{name}: {value}")

        # Build raw response headers
        raw = "\r\n".join(lines).encode("utf-8")
        raw += b"\r\n\r\n"

        return raw
