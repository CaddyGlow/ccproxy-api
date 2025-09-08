"""Service for detecting Codex CLI using centralized detection."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request, Response

from ccproxy.config.settings import Settings
from ccproxy.config.utils import get_ccproxy_cache_dir
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.utils.caching import async_ttl_cache
from ccproxy.utils.headers import HeaderBag

from .models import CodexCacheData, CodexHeaders, CodexInstructionsData


logger = get_plugin_logger()


if TYPE_CHECKING:
    from .config import CodexSettings
    from .models import CodexCliInfo


class CodexDetectionService:
    """Service for automatically detecting Codex CLI headers at startup."""

    def __init__(
        self,
        settings: Settings,
        cli_service: CLIDetectionService | None = None,
        codex_settings: CodexSettings | None = None,
    ) -> None:
        """Initialize Codex detection service.

        Args:
            settings: Application settings
            cli_service: Optional CLI detection service for dependency injection.
                        If None, creates its own instance.
            codex_settings: Optional Codex plugin settings for plugin-specific configuration.
                           If None, uses default configuration.
        """
        self.settings = settings
        self.codex_settings = codex_settings
        self.cache_dir = get_ccproxy_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached_data: CodexCacheData | None = None
        self._cli_service = cli_service or CLIDetectionService(settings)
        self._cli_info: CodexCliInfo | None = None

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

    def get_cli_health_info(self) -> CodexCliInfo:
        """Get lightweight CLI health info using centralized detection, cached locally.

        Returns:
            CodexCliInfo with availability, version, and binary path
        """
        from .models import CodexCliInfo, CodexCliStatus

        if self._cli_info is not None:
            return self._cli_info

        info = self._cli_service.get_cli_info("codex")
        status = (
            CodexCliStatus.AVAILABLE
            if info["is_available"]
            else CodexCliStatus.NOT_INSTALLED
        )
        cli_info = CodexCliInfo(
            status=status,
            version=info.get("version"),
            binary_path=info.get("path"),
        )
        self._cli_info = cli_info
        return cli_info

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
            # Preserve order and original casing when capturing headers
            bag = HeaderBag.from_request(request, case_mode="preserve")
            captured_data["headers_ordered"] = list(bag.items())
            captured_data["headers"] = bag.to_dict()

            # Capture raw body
            raw_body = await request.body()
            captured_data["body"] = raw_body

            # Build complete JSON request structure
            full_request: dict[str, Any] = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params)
                if request.query_params
                else {},
                "headers_ordered": list(bag.items()),
                "headers": bag.to_dict(),
            }

            # Parse body as JSON if possible, otherwise store as string
            try:
                if raw_body:
                    body_json = json.loads(raw_body.decode("utf-8"))
                    full_request["body_json"] = body_json
                    full_request["body_raw"] = raw_body.decode("utf-8")
                else:
                    full_request["body_json"] = None
                    full_request["body_raw"] = ""
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.debug("body_parsing_failed", error=str(e), category="plugin")
                full_request["body_json"] = None
                full_request["body_raw"] = raw_body.hex() if raw_body else ""
                full_request["body_parse_error"] = str(e)

            # Store complete request structure
            captured_data["full_request_json"] = full_request

            logger.debug(
                "request_captured",
                method=request.method,
                path=request.url.path,
                headers_count=len(list(bag.items())),
                body_size=len(raw_body),
                category="plugin",
            )

            # Return a mock response to satisfy Codex CLI
            return Response(
                content='{"choices": [{"message": {"content": "Test response"}}]}',
                media_type="application/json",
                status_code=200,
            )

        # Create temporary FastAPI app
        temp_app = FastAPI()
        # Current Codex endpoint used by CLI
        temp_app.post("/backend-api/codex/responses")(capture_handler)

        # from starlette.middleware.base import BaseHTTPMiddleware
        # from starlette.requests import Request
        #
        # Another way to recover the headers
        # class DumpHeadersMiddleware(BaseHTTPMiddleware):
        #     async def dispatch(self, request: Request, call_next):
        #         # Print all headers
        #         print("Request Headers:")
        #         for name, value in request.headers.items():
        #             print(f"{name}: {value}")
        #         response = await call_next(request)
        #         return response
        #
        # temp_app.add_middleware(DumpHeadersMiddleware)

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

            # Determine home directory mode based on configuration
            home_dir = os.environ.get("HOME")
            temp_context = None
            if (
                self.codex_settings
                and self.codex_settings.detection_home_mode == "temp"
            ):
                temp_context = tempfile.TemporaryDirectory()
                home_dir = temp_context.__enter__()
                logger.debug(
                    "using_temporary_home_directory",
                    home_dir=home_dir,
                    category="plugin",
                )
            else:
                logger.debug(
                    "using_actual_home_directory", home_dir=home_dir, category="plugin"
                )

            try:
                # Execute Codex CLI with proxy
                env: dict[str, str] = dict(os.environ)
                env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/backend-api/codex"
                env["OPENAI_API_KEY"] = "dummy-key-for-detection"
                if home_dir is not None:
                    env["HOME"] = home_dir
                del env["OPENAI_API_KEY"]

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

            finally:
                # Clean up temporary directory if used
                if temp_context is not None:
                    temp_context.__exit__(None, None, None)

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
                codex_version=version,
                headers=headers,
                instructions=instructions,
                raw_headers_ordered=captured_data.get("headers_ordered", []),
                full_request_json=captured_data.get("full_request_json"),
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
        self._cli_info = None
        logger.debug("detection_cache_cleared", category="plugin")
