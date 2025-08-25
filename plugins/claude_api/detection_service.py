"""Claude API plugin detection service using centralized detection."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response

from ccproxy.config.discovery import get_ccproxy_cache_dir
from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.detection import (
    ClaudeCacheData,
    ClaudeCodeHeaders,
    SystemPromptData,
)
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.utils.caching import async_ttl_cache


logger = get_plugin_logger()


class ClaudeAPIDetectionService:
    """Claude API plugin detection service for automatically detecting Claude CLI headers."""

    def __init__(
        self, settings: Settings, cli_service: CLIDetectionService | None = None
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

    async def initialize_detection(self) -> ClaudeCacheData:
        """Initialize Claude detection at startup."""
        try:
            # Get current Claude version
            current_version = await self._get_claude_version()

            # Try to load from cache first
            detected_data = self._load_from_cache(current_version)
            cached = detected_data is not None
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
            captured_data["headers"] = dict(request.headers)
            captured_data["body"] = await request.body()
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

            # Extract headers and system prompt
            headers = self._extract_headers(captured_data["headers"])
            system_prompt = self._extract_system_prompt(captured_data["body"])

            return ClaudeCacheData(
                claude_version=version, headers=headers, system_prompt=system_prompt
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

        try:
            with cache_file.open("r") as f:
                data = json.load(f)
                return ClaudeCacheData.model_validate(data)
        except Exception:
            return None

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

    def _extract_headers(self, headers: dict[str, str]) -> ClaudeCodeHeaders:
        """Extract Claude CLI headers from captured request."""
        try:
            return ClaudeCodeHeaders.model_validate(headers)
        except Exception as e:
            logger.error("header_extraction_failed", error=str(e), category="plugin")
            raise ValueError(f"Failed to extract required headers: {e}") from e

    def _extract_system_prompt(self, body: bytes) -> SystemPromptData:
        """Extract system prompt from captured request body."""
        try:
            data = json.loads(body.decode("utf-8"))
            system_content = data.get("system")

            if system_content is None:
                raise ValueError("No system field found in request body")

            return SystemPromptData(system_field=system_content)

        except Exception as e:
            logger.error(
                "system_prompt_extraction_failed", error=str(e), category="plugin"
            )
            raise ValueError(f"Failed to extract system prompt: {e}") from e

    def _get_fallback_data(self) -> ClaudeCacheData:
        """Get fallback data when detection fails."""
        logger.warning("using_fallback_claude_data", category="plugin")

        # Load fallback data from package data file
        package_data_file = (
            Path(__file__).parent / "data" / "claude_headers_fallback.json"
        )
        with package_data_file.open("r") as f:
            fallback_data_dict = json.load(f)
            return ClaudeCacheData.model_validate(fallback_data_dict)

    def invalidate_cache(self) -> None:
        """Clear all cached detection data."""
        # Clear the async cache for _get_claude_version
        if hasattr(self._get_claude_version, "cache_clear"):
            self._get_claude_version.cache_clear()
        logger.debug("detection_cache_cleared", category="plugin")
