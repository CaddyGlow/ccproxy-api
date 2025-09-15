"""Claude API plugin detection service using centralized detection."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response

from ccproxy.config.settings import Settings
from ccproxy.config.utils import get_ccproxy_cache_dir
from ccproxy.core.logging import get_plugin_logger
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

            # TODO: add proper testing without claude cli installed
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

        server_task = asyncio.create_task(server.serve())

        try:
            # Wait for server to start
            await asyncio.sleep(0.5)

            # Execute Claude CLI with proxy
            env = {**dict(os.environ), "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}"}

            # Get claude command from CLI service
            cli_info = self._cli_service.get_cli_info("claude")
            if not cli_info["is_available"] or not cli_info["command"]:
                raise FileNotFoundError("Claude CLI not found for header detection")

            # Prepare command
            cmd = cli_info["command"] + ["test"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for process with timeout
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except TimeoutError:
                process.kill()
                await process.wait()

            # Stop server
            server.should_exit = True
            await server_task

            if not captured_data:
                raise RuntimeError("Failed to capture Claude CLI request")

            # Sanitize headers/body for cache
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

            return ClaudeCacheData(
                claude_version=version,
                headers=headers_dict,
                body_json=body_json,
                method=captured_data.get("method"),
                url=captured_data.get("url"),
                path=captured_data.get("path"),
                query_params=captured_data.get("query_params"),
            )

        except Exception as e:
            # Ensure server is stopped
            server.should_exit = True
            if not server_task.done():
                await server_task
            raise

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

    def get_system_prompt(self, mode: str = "minimal") -> dict[str, Any]:
        """Return a system prompt dict for injection based on cached body_json.

        mode: "none", "minimal", or "full"
        """
        data = self.get_cached_data()
        if not data or not data.body_json:
            return {}
        system_value = data.body_json.get("system")
        if system_value is None:
            return {}
        if mode == "none":
            return {}
        if mode == "minimal" and isinstance(system_value, list):
            if len(system_value) > 0:
                return {"system": [system_value[0]]}
            return {}
        # full or non-list
        return {"system": system_value}
