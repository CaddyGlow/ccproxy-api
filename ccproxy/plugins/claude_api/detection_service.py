"""Claude API plugin detection service using centralized detection."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response

from ccproxy.config.settings import Settings
from ccproxy.config.utils import get_ccproxy_cache_dir
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.utils.caching import async_ttl_cache
from ccproxy.utils.headers import extract_request_headers

from .models import ClaudeCacheData


logger = get_plugin_logger()


if TYPE_CHECKING:
    from .models import ClaudeCliInfo


class ClaudeAPIDetectionService:
    """Claude API plugin detection service for automatically detecting Claude CLI headers."""

    # Headers to ignore at injection time (lowercase). Cache keeps keys (possibly empty) to preserve order.
    ignores_header: list[str] = [
        # Common excludes
        "host",
        "content-length",
        "authorization",
        "x-api-key",
    ]

    redact_headers: list[str] = [
        "x-api-key",
        "authorization",
    ]

    def __init__(
        self,
        settings: Settings,
        cli_service: CLIDetectionService | None = None,
        redact_sensitive_cache: bool = True,
    ) -> None:
        """Initialize Claude detection service.

        Args:
            settings: Application settings
            cli_service: Optional CLIDetectionService instance for dependency injection.
                        If None, creates a new instance for backward compatibility.
        """
        self.settings = settings
        self.cache_dir = get_ccproxy_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_data: ClaudeCacheData | None = None
        self._cli_service = cli_service or CLIDetectionService(settings)
        self._cli_info: ClaudeCliInfo | None = None
        self._redact_sensitive_cache = redact_sensitive_cache

    async def initialize_detection(self) -> ClaudeCacheData:
        """Initialize Claude detection at startup."""
        try:
            # Get current Claude version
            current_version = await self._get_claude_version()

            # Try to load from cache first
            cached = False
            try:
                detected_data = self._load_from_cache(current_version)
                cached = detected_data is not None

            except Exception as e:
                logger.warning(
                    "invalid_cache_file",
                    error=str(e),
                    category="plugin",
                    exc_info=e,
                )

            if not cached:
                # No cache or version changed - detect fresh
                detected_data = await self._detect_claude_headers(current_version)
                # Cache the results
                self._save_to_cache(detected_data)

            self._cached_data = detected_data

            logger.trace(
                "detection_headers_completed",
                version=current_version,
                cached=cached,
            )

            if detected_data is None:
                raise ValueError("Claude detection failed")
            return detected_data

        except Exception as e:
            logger.warning(
                "detection_claude_headers_failed",
                fallback=True,
                error=e,
                category="plugin",
            )
            # Return fallback data
            fallback_data = self._get_fallback_data()
            self._cached_data = fallback_data
            return fallback_data

    def get_cached_data(self) -> ClaudeCacheData | None:
        """Get currently cached detection data."""
        return self._cached_data

    def get_detected_headers(self) -> DetectedHeaders:
        """Return cached headers as structured data."""

        data = self.get_cached_data()
        if not data:
            return DetectedHeaders()
        return data.headers

    def get_detected_prompts(self) -> DetectedPrompts:
        """Return cached prompt metadata as structured data."""

        data = self.get_cached_data()
        if not data:
            return DetectedPrompts()
        return data.prompts

    def get_ignored_headers(self) -> list[str]:
        """Headers that should be ignored when injecting CLI values."""

        return list(self.ignores_header)

    def get_redacted_headers(self) -> list[str]:
        """Headers that must never be forwarded from detection cache."""

        return list(self.redact_headers)

    def get_cli_health_info(self) -> ClaudeCliInfo:
        """Get lightweight CLI health info using centralized detection, cached locally.

        Returns:
            ClaudeCliInfo with availability, version, and binary path
        """
        from .models import ClaudeCliInfo, ClaudeCliStatus

        if self._cli_info is not None:
            return self._cli_info

        info = self._cli_service.get_cli_info("claude")
        status = (
            ClaudeCliStatus.AVAILABLE
            if info["is_available"]
            else ClaudeCliStatus.NOT_INSTALLED
        )
        cli_info = ClaudeCliInfo(
            status=status,
            version=info.get("version"),
            binary_path=info.get("path"),
        )
        self._cli_info = cli_info
        return cli_info

    def get_version(self) -> str | None:
        """Get the detected Claude CLI version."""
        if self._cached_data:
            return self._cached_data.claude_version
        return None

    def get_cli_path(self) -> list[str] | None:
        """Get the Claude CLI command with caching.

        Returns:
            Command list to execute Claude CLI if found, None otherwise
        """
        info = self._cli_service.get_cli_info("claude")
        return info["command"] if info["is_available"] else None

    def get_binary_path(self) -> list[str] | None:
        """Alias for get_cli_path for consistency with Codex."""
        return self.get_cli_path()

    @async_ttl_cache(maxsize=16, ttl=900.0)  # 15 minute cache for version
    async def _get_claude_version(self) -> str:
        """Get Claude CLI version with caching."""
        try:
            # Use centralized CLI detection
            result = await self._cli_service.detect_cli(
                binary_name="claude",
                package_name="@anthropic-ai/claude-code",
                version_flag="--version",
                cache_key="claude_api_version",
            )

            if result.is_available and result.version:
                return result.version
            else:
                raise FileNotFoundError("Claude CLI not found")

        except Exception as e:
            logger.warning(
                "claude_version_detection_failed", error=str(e), category="plugin"
            )
            return "unknown"

    async def _detect_claude_headers(self, version: str) -> ClaudeCacheData:
        """Execute Claude CLI with proxy to capture headers and system prompt."""
        # Data captured from the request
        captured_data: dict[str, Any] = {}

        async def capture_handler(request: Request) -> Response:
            """Capture the Claude CLI request."""
            # Capture request details
            headers = extract_request_headers(request)
            captured_data["headers"] = headers
            captured_data["method"] = request.method
            captured_data["url"] = str(request.url)
            captured_data["path"] = request.url.path
            captured_data["query_params"] = (
                dict(request.query_params) if request.query_params else {}
            )

            raw_body = await request.body()
            captured_data["body"] = raw_body
            # Try to parse to JSON for body_json
            try:
                captured_data["body_json"] = (
                    json.loads(raw_body.decode("utf-8")) if raw_body else None
                )
            except Exception:
                captured_data["body_json"] = None
            # Return a mock response to satisfy Claude CLI
            return Response(
                content='{"type": "message", "content": [{"type": "text", "text": "Test response"}]}',
                media_type="application/json",
                status_code=200,
            )

        # Create temporary FastAPI app
        temp_app = FastAPI()
        temp_app.post("/v1/messages")(capture_handler)

        # Find available port
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        # Start server in background
        from uvicorn import Config, Server

        config = Config(temp_app, host="127.0.0.1", port=port, log_level="error")
        server = Server(config)

        server_ready = asyncio.Event()

        @temp_app.on_event("startup")
        async def signal_server_ready() -> None:
            """Signal when the temporary detection server starts."""

            server_ready.set()

        server_task = asyncio.create_task(server.serve())
        ready_task = asyncio.create_task(server_ready.wait())

        try:
            done, _pending = await asyncio.wait(
                {ready_task, server_task},
                timeout=5,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if ready_task in done:
                await ready_task
            elif server_task in done:
                await server_task
                raise RuntimeError(
                    "Claude detection server exited before signalling readiness"
                )
            else:
                raise TimeoutError(
                    "Timed out waiting for Claude detection server startup"
                )

            stdout, stderr = b"", b""

            env: dict[str, str] = dict(os.environ)
            env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

            home_path = os.environ.get("HOME")
            cwd_path = Path(home_path) if home_path else Path.cwd()

            logger.debug(
                "detection_service_using",
                home_dir=home_path,
                cwd=cwd_path,
                category="plugin",
            )

            if home_path is not None:
                env["HOME"] = home_path

            cli_info = self._cli_service.get_cli_info("claude")
            if not cli_info["is_available"] or not cli_info["command"]:
                raise FileNotFoundError("Claude CLI not found for header detection")

            cmd = cli_info["command"] + ["test"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd_path),
            )

            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()

            stdout = await process.stdout.read() if process.stdout else b""
            stderr = await process.stderr.read() if process.stderr else b""
        finally:
            if not ready_task.done():
                ready_task.cancel()
                with suppress(asyncio.CancelledError):
                    await ready_task

            server.should_exit = True
            await server_task

        if not captured_data:
            logger.error(
                "failed_to_capture_claude_cli_request",
                stdout=stdout.decode(errors="ignore"),
                stderr=stderr.decode(errors="ignore"),
                category="plugin",
            )
            raise RuntimeError("Failed to capture Claude CLI request")

        headers_dict = (
            self._sanitize_headers_for_cache(captured_data["headers"])
            if self._redact_sensitive_cache
            else captured_data["headers"]
        )
        body_json = (
            self._sanitize_body_json_for_cache(captured_data.get("body_json"))
            if self._redact_sensitive_cache
            else captured_data.get("body_json")
        )

        prompts = DetectedPrompts.from_body(body_json)

        return ClaudeCacheData(
            claude_version=version,
            headers=DetectedHeaders(headers_dict),
            prompts=prompts,
            body_json=body_json,
            method=captured_data.get("method"),
            url=captured_data.get("url"),
            path=captured_data.get("path"),
            query_params=captured_data.get("query_params"),
        )

    def _load_from_cache(self, version: str) -> ClaudeCacheData | None:
        """Load cached data for specific Claude version."""
        cache_file = self.cache_dir / f"claude_headers_{version}.json"

        if not cache_file.exists():
            return None

        with cache_file.open("r") as f:
            data = json.load(f)
            return ClaudeCacheData.model_validate(data)

    def _save_to_cache(self, data: ClaudeCacheData) -> None:
        """Save detection data to cache."""
        cache_file = self.cache_dir / f"claude_headers_{data.claude_version}.json"

        try:
            with cache_file.open("w") as f:
                json.dump(data.model_dump(), f, indent=2, default=str)
            logger.debug(
                "cache_saved",
                file=str(cache_file),
                version=data.claude_version,
                category="plugin",
            )
        except Exception as e:
            logger.warning(
                "cache_save_failed",
                file=str(cache_file),
                error=str(e),
                category="plugin",
            )

    def _get_fallback_data(self) -> ClaudeCacheData:
        """Get fallback data when detection fails."""
        logger.warning("using_fallback_claude_data", category="plugin")

        # Load fallback data from package data file
        package_data_file = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "claude_headers_fallback.json"
        )
        with package_data_file.open("r") as f:
            fallback_data_dict = json.load(f)
            return ClaudeCacheData.model_validate(fallback_data_dict)

    def invalidate_cache(self) -> None:
        """Clear all cached detection data."""
        # Clear the async cache for _get_claude_version
        if hasattr(self._get_claude_version, "cache_clear"):
            self._get_claude_version.cache_clear()
        # Clear CLI info cache
        self._cli_info = None
        logger.debug("detection_cache_cleared", category="plugin")

    # --- Helpers ---
    def _sanitize_headers_for_cache(self, headers: dict[str, str]) -> dict[str, str]:
        """Redact sensitive headers for cache while preserving keys and order."""
        # Build ordered dict copy
        sanitized: dict[str, str] = {}
        for k, v in headers.items():
            lk = k.lower()
            if lk in {"authorization", "host"}:
                sanitized[lk] = ""
            else:
                sanitized[lk] = v
        return sanitized

    def _sanitize_body_json_for_cache(
        self, body: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if body is None:
            return None
        # For Claude, no specific fields to redact currently; return as-is
        return body

    def get_system_prompt(self, mode: str | None = "minimal") -> dict[str, Any]:
        """Return a system prompt dict for injection based on cached prompts.

        mode: "none", "minimal", or "full"
        """
        prompts = self.get_detected_prompts()
        mode_value = "full" if mode is None else mode
        return prompts.system_payload(mode=mode_value)
