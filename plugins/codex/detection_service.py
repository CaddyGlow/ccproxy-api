"""Service for detecting Codex CLI using centralized detection."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response

from ccproxy.config.discovery import get_ccproxy_cache_dir
from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.detection import (
    CodexCacheData,
    CodexHeaders,
    CodexInstructionsData,
)
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.utils.caching import async_ttl_cache


logger = get_plugin_logger()


class CodexDetectionService:
    """Service for automatically detecting Codex CLI headers at startup."""

    def __init__(
        self, settings: Settings, cli_service: CLIDetectionService | None = None
    ) -> None:
        """Initialize Codex detection service.

        Args:
            settings: Application settings
            cli_service: Optional CLI detection service for dependency injection.
                        If None, creates its own instance.
        """
        self.settings = settings
        self.cache_dir = get_ccproxy_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_data: CodexCacheData | None = None
        self._cli_service = cli_service or CLIDetectionService(settings)

    async def initialize_detection(self) -> CodexCacheData:
        """Initialize Codex detection at startup."""
        try:
            # Get current Codex version
            current_version = await self._get_codex_version()

            # Try to load from cache first
            detected_data = self._load_from_cache(current_version)
            cached = detected_data is not None
            if not cached:
                # No cache or version changed - detect fresh
                detected_data = await self._detect_codex_headers(current_version)
                # Cache the results
                self._save_to_cache(detected_data)

            self._cached_data = detected_data

            logger.trace(
                "detection_headers_completed",
                version=current_version,
                cached=cached,
            )

            # TODO: add proper testing without codex cli installed
            if detected_data is None:
                raise ValueError("Codex detection failed")
            return detected_data

        except Exception as e:
            logger.warning(
                "detection_codex_headers_failed",
                fallback=True,
                exc_info=e,
                category="plugin",
            )
            # Return fallback data
            fallback_data = self._get_fallback_data()
            self._cached_data = fallback_data
            return fallback_data

    def get_cached_data(self) -> CodexCacheData | None:
        """Get currently cached detection data."""
        return self._cached_data

    def get_version(self) -> str:
        """Get the Codex CLI version.

        Returns:
            Version string or "unknown" if not available
        """
        data = self.get_cached_data()
        return data.codex_version if data else "unknown"

    def get_cli_path(self) -> list[str] | None:
        """Get the Codex CLI command with caching.

        Returns:
            Command list to execute Codex CLI if found, None otherwise
        """
        info = self._cli_service.get_cli_info("codex")
        return info["command"] if info["is_available"] else None

    def get_binary_path(self) -> list[str] | None:
        """Alias for get_cli_path for backward compatibility."""
        return self.get_cli_path()

    @async_ttl_cache(maxsize=16, ttl=900.0)  # 15 minute cache for version
    async def _get_codex_version(self) -> str:
        """Get Codex CLI version with caching."""
        try:
            # Custom parser for Codex version format
            def parse_codex_version(output: str) -> str:
                # Handle "codex 0.21.0" format
                if " " in output:
                    return output.split()[-1]
                return output

            # Use centralized CLI detection
            result = await self._cli_service.detect_cli(
                binary_name="codex",
                package_name="@openai/codex",
                version_flag="--version",
                version_parser=parse_codex_version,
                cache_key="codex_version",
            )

            if result.is_available and result.version:
                return result.version
            else:
                raise FileNotFoundError("Codex CLI not found")

        except Exception as e:
            logger.warning(
                "codex_version_detection_failed", error=str(e), category="plugin"
            )
            return "unknown"

    async def _detect_codex_headers(self, version: str) -> CodexCacheData:
        """Execute Codex CLI with proxy to capture headers and instructions."""
        # Data captured from the request
        captured_data: dict[str, Any] = {}

        async def capture_handler(request: Request) -> Response:
            """Capture the Codex CLI request."""
            captured_data["headers"] = dict(request.headers)
            captured_data["body"] = await request.body()
            # Return a mock response to satisfy Codex CLI
            return Response(
                content='{"choices": [{"message": {"content": "Test response"}}]}',
                media_type="application/json",
                status_code=200,
            )

        # Create temporary FastAPI app
        temp_app = FastAPI()
        temp_app.post("/backend-api/codex/responses")(capture_handler)

        # Find available port
        sock = socket.socket()
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()

        # Start server in background
        from uvicorn import Config, Server

        config = Config(temp_app, host="127.0.0.1", port=port, log_level="error")
        server = Server(config)

        logger.debug("start", category="plugin")
        server_task = asyncio.create_task(server.serve())

        try:
            # Wait for server to start
            await asyncio.sleep(0.5)

            stdout, stderr = b"", b""
            with tempfile.TemporaryDirectory() as temp_home:
                # Execute Codex CLI with proxy
                env = {
                    **dict(os.environ),
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/backend-api/codex",
                    "OPENAI_API_KEY": "dummy-key-for-detection",
                    "HOME": temp_home,
                }

                # Get codex command from CLI service
                cli_info = self._cli_service.get_cli_info("codex")
                if not cli_info["is_available"] or not cli_info["command"]:
                    raise FileNotFoundError("Codex CLI not found for header detection")

                # Prepare command
                cmd = cli_info["command"] + ["exec", "test"]

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                # Wait for process with timeout
                try:
                    await asyncio.wait_for(process.wait(), timeout=300)
                except TimeoutError:
                    process.kill()
                    await process.wait()

                stdout = await process.stdout.read() if process.stdout else b""
                stderr = await process.stderr.read() if process.stderr else b""

            # Stop server
            server.should_exit = True
            await server_task

            if not captured_data:
                logger.error(
                    "failed_to_capture_codex_cli_request",
                    stdout=stdout.decode(errors="ignore"),
                    stderr=stderr.decode(errors="ignore"),
                    category="plugin",
                )
                raise RuntimeError("Failed to capture Codex CLI request")

            # Extract headers and instructions
            headers = self._extract_headers(captured_data["headers"])
            instructions = self._extract_instructions(captured_data["body"])

            return CodexCacheData(
                codex_version=version, headers=headers, instructions=instructions
            )

        except Exception as e:
            # Ensure server is stopped
            server.should_exit = True
            if not server_task.done():
                await server_task
            raise

    def _load_from_cache(self, version: str) -> CodexCacheData | None:
        """Load cached data for specific Codex version."""
        cache_file = self.cache_dir / f"codex_headers_{version}.json"

        if not cache_file.exists():
            return None

        try:
            with cache_file.open("r") as f:
                data = json.load(f)
                return CodexCacheData.model_validate(data)
        except Exception:
            return None

    def _save_to_cache(self, data: CodexCacheData) -> None:
        """Save detection data to cache."""
        cache_file = self.cache_dir / f"codex_headers_{data.codex_version}.json"

        try:
            with cache_file.open("w") as f:
                json.dump(data.model_dump(), f, indent=2, default=str)
            logger.debug(
                "cache_saved",
                file=str(cache_file),
                version=data.codex_version,
                category="plugin",
            )
        except Exception as e:
            logger.warning(
                "cache_save_failed",
                file=str(cache_file),
                error=str(e),
                category="plugin",
            )

    def _extract_headers(self, headers: dict[str, str]) -> CodexHeaders:
        """Extract Codex CLI headers from captured request."""
        try:
            return CodexHeaders.model_validate(headers)
        except Exception as e:
            logger.error("header_extraction_failed", error=str(e), category="plugin")
            raise ValueError(f"Failed to extract required headers: {e}") from e

    def _extract_instructions(self, body: bytes) -> CodexInstructionsData:
        """Extract instructions from captured request body."""
        try:
            data = json.loads(body.decode("utf-8"))
            instructions_content = data.get("instructions")

            if instructions_content is None:
                raise ValueError("No instructions field found in request body")

            return CodexInstructionsData(instructions_field=instructions_content)

        except Exception as e:
            logger.error(
                "instructions_extraction_failed", error=str(e), category="plugin"
            )
            raise ValueError(f"Failed to extract instructions: {e}") from e

    def _get_fallback_data(self) -> CodexCacheData:
        """Get fallback data when detection fails."""
        logger.warning("using_fallback_codex_data", category="plugin")

        # Load fallback data from package data file
        package_data_file = (
            Path(__file__).parent / "data" / "codex_headers_fallback.json"
        )
        with package_data_file.open("r") as f:
            fallback_data_dict = json.load(f)
            return CodexCacheData.model_validate(fallback_data_dict)

    def invalidate_cache(self) -> None:
        """Clear all cached detection data."""
        # Clear the async cache for _get_codex_version
        if hasattr(self._get_codex_version, "cache_clear"):
            self._get_codex_version.cache_clear()
        logger.debug("detection_cache_cleared", category="plugin")
