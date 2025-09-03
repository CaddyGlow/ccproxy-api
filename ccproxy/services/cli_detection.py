"""Centralized CLI detection service for all plugins.

This module provides a unified interface for detecting CLI binaries,
checking versions, and managing CLI-related state across all plugins.
It eliminates duplicate CLI detection logic by consolidating common patterns.
"""

import asyncio
import json
from typing import Any, NamedTuple

import structlog

from ccproxy.config.discovery import get_ccproxy_cache_dir
from ccproxy.config.settings import Settings
from ccproxy.utils.binary_resolver import BinaryResolver, CLIInfo
from ccproxy.utils.caching import TTLCache


logger = structlog.get_logger(__name__)


class CLIDetectionResult(NamedTuple):
    """Result of CLI detection for a specific binary."""

    name: str
    version: str | None
    command: list[str] | None
    is_available: bool
    source: str  # "path", "package_manager", "fallback", or "unknown"
    package_manager: str | None = None
    cached: bool = False
    fallback_data: dict[str, Any] | None = None


class CLIDetectionService:
    """Centralized service for CLI detection across all plugins.

    This service provides:
    - Unified binary detection using BinaryResolver
    - Version detection with caching
    - Fallback data support for when CLI is not available
    - Consistent logging and error handling
    """

    def __init__(
        self, settings: Settings, binary_resolver: BinaryResolver | None = None
    ) -> None:
        """Initialize the CLI detection service.

        Args:
            settings: Application settings
            binary_resolver: Optional binary resolver instance. If None, creates a new one.
        """
        self.settings = settings
        self.cache_dir = get_ccproxy_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use injected resolver or create from settings for backward compatibility
        self.resolver = binary_resolver or BinaryResolver.from_settings(settings)

        # Enhanced TTL cache for detection results (10 minute TTL)
        self._detection_cache = TTLCache(maxsize=64, ttl=600.0)

        # Separate cache for version info (longer TTL since versions change infrequently)
        self._version_cache = TTLCache(maxsize=32, ttl=1800.0)  # 30 minutes

    async def detect_cli(
        self,
        binary_name: str,
        package_name: str | None = None,
        version_flag: str = "--version",
        version_parser: Any | None = None,
        fallback_data: dict[str, Any] | None = None,
        cache_key: str | None = None,
    ) -> CLIDetectionResult:
        """Detect a CLI binary and its version.

        Args:
            binary_name: Name of the binary to detect (e.g., "claude", "codex")
            package_name: NPM package name if different from binary name
            version_flag: Flag to get version (default: "--version")
            version_parser: Optional callable to parse version output
            fallback_data: Optional fallback data if CLI is not available
            cache_key: Optional cache key (defaults to binary_name)

        Returns:
            CLIDetectionResult with detection information
        """
        cache_key = cache_key or binary_name

        # Check TTL cache first
        cached_result = self._detection_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(
                "cli_detection_cached",
                binary=binary_name,
                version=cached_result.version,
                available=cached_result.is_available,
                cache_hit=True,
            )
            return cached_result  # type: ignore[no-any-return]

        # Try to detect the binary
        result = self.resolver.find_binary(binary_name, package_name)

        if result:
            # Binary found - get version
            version = await self._get_cli_version(
                result.command, version_flag, version_parser
            )

            # Determine source
            source = "path" if result.is_direct else "package_manager"

            detection_result = CLIDetectionResult(
                name=binary_name,
                version=version,
                command=result.command,
                is_available=True,
                source=source,
                package_manager=result.package_manager,
                cached=False,
            )

            logger.debug(
                "cli_detection_success",
                binary=binary_name,
                version=version,
                source=source,
                package_manager=result.package_manager,
                command=result.command,
                cached=cached_result is not None,
            )

        elif fallback_data:
            # Use fallback data
            detection_result = CLIDetectionResult(
                name=binary_name,
                version=fallback_data.get("version", "unknown"),
                command=None,
                is_available=False,
                source="fallback",
                package_manager=None,
                cached=False,
                fallback_data=fallback_data,
            )

            logger.warning(
                "cli_detection_using_fallback",
                binary=binary_name,
                reason="CLI not found",
            )

        else:
            # Not found and no fallback
            detection_result = CLIDetectionResult(
                name=binary_name,
                version=None,
                command=None,
                is_available=False,
                source="unknown",
                package_manager=None,
                cached=False,
            )

            logger.error(
                "cli_detection_failed",
                binary=binary_name,
                package=package_name,
            )

        # Cache the result with TTL
        self._detection_cache.set(cache_key, detection_result)

        return detection_result

    async def _get_cli_version(
        self,
        cli_command: list[str],
        version_flag: str,
        version_parser: Any | None = None,
    ) -> str | None:
        """Get CLI version by executing version command with caching.

        Args:
            cli_command: Command list to execute CLI
            version_flag: Flag to get version
            version_parser: Optional callable to parse version output

        Returns:
            Version string if successful, None otherwise
        """
        # Create cache key from command and flag
        cache_key = f"version:{':'.join(cli_command)}:{version_flag}"

        # Check version cache first (longer TTL since versions change infrequently)
        cached_version = self._version_cache.get(cache_key)
        if cached_version is not None:
            logger.debug(
                "cli_version_cached",
                command=cli_command[0],
                version=cached_version,
                cache_hit=True,
            )
            return cached_version  # type: ignore[no-any-return]

        try:
            # Prepare command with version flag
            cmd = cli_command + [version_flag]

            # Run command with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)

            version = None
            if process.returncode == 0 and stdout:
                version_output = stdout.decode().strip()

                # Use custom parser if provided
                if version_parser:
                    parsed = version_parser(version_output)
                    version = str(parsed) if parsed is not None else None
                else:
                    # Default parsing logic
                    version = self._parse_version_output(version_output)

            # Try stderr as some CLIs output version there
            if not version and stderr:
                version_output = stderr.decode().strip()
                if version_parser:
                    parsed = version_parser(version_output)
                    version = str(parsed) if parsed is not None else None
                else:
                    version = self._parse_version_output(version_output)

            # Cache the version result (even if None)
            self._version_cache.set(cache_key, version)

            return version

        except TimeoutError:
            logger.debug("cli_version_timeout", command=cli_command)
            # Cache timeout result briefly to avoid repeated attempts
            self._version_cache.set(cache_key, None)
            return None
        except Exception as e:
            logger.debug("cli_version_error", command=cli_command, error=str(e))
            # Cache error result briefly to avoid repeated attempts
            self._version_cache.set(cache_key, None)
            return None

    def _parse_version_output(self, output: str) -> str:
        """Parse version from CLI output using common patterns.

        Args:
            output: Raw version command output

        Returns:
            Parsed version string
        """
        # Handle various common formats
        if "/" in output:
            # Handle "tool/1.0.0" format
            output = output.split("/")[-1]

        if "(" in output:
            # Handle "1.0.0 (Tool Name)" format
            output = output.split("(")[0].strip()

        # Extract version number pattern (e.g., "1.0.0", "v1.0.0")
        import re

        version_pattern = r"v?(\d+\.\d+(?:\.\d+)?(?:-[\w.]+)?)"
        match = re.search(version_pattern, output)
        if match:
            return match.group(1)

        # Return cleaned output if no pattern matches
        return output.strip()

    def load_cached_version(
        self, binary_name: str, cache_file: str | None = None
    ) -> str | None:
        """Load cached version for a binary.

        Args:
            binary_name: Name of the binary
            cache_file: Optional cache file name

        Returns:
            Cached version string or None
        """
        cache_file_name = cache_file or f"{binary_name}_version.json"
        cache_path = self.cache_dir / cache_file_name

        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r") as f:
                data = json.load(f)
                version = data.get("version")
                return str(version) if version is not None else None
        except Exception as e:
            logger.debug("cache_load_error", file=str(cache_path), error=str(e))
            return None

    def save_cached_version(
        self,
        binary_name: str,
        version: str,
        cache_file: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> None:
        """Save version to cache.

        Args:
            binary_name: Name of the binary
            version: Version string to cache
            cache_file: Optional cache file name
            additional_data: Additional data to cache
        """
        cache_file_name = cache_file or f"{binary_name}_version.json"
        cache_path = self.cache_dir / cache_file_name

        try:
            data = {"binary": binary_name, "version": version}
            if additional_data:
                data.update(additional_data)

            with cache_path.open("w") as f:
                json.dump(data, f, indent=2)

            logger.debug("cache_saved", file=str(cache_path), version=version)
        except Exception as e:
            logger.warning("cache_save_error", file=str(cache_path), error=str(e))

    def get_cli_info(self, binary_name: str) -> CLIInfo:
        """Get CLI information in standard format.

        Args:
            binary_name: Name of the binary

        Returns:
            CLIInfo dictionary with structured information
        """
        # Check if we have cached detection result
        cached_result = self._detection_cache.get(binary_name)
        if cached_result is not None:
            return CLIInfo(
                name=cached_result.name,
                version=cached_result.version,
                source=cached_result.source,
                path=cached_result.command[0] if cached_result.command else None,
                command=cached_result.command or [],
                package_manager=cached_result.package_manager,
                is_available=cached_result.is_available,
            )

        # Fall back to resolver
        return self.resolver.get_cli_info(binary_name)

    def clear_cache(self) -> None:
        """Clear all detection caches."""
        self._detection_cache.clear()
        self._version_cache.clear()
        self.resolver.clear_cache()
        logger.debug("cli_detection_cache_cleared")

    def get_all_detected(self) -> dict[str, CLIDetectionResult]:
        """Get all detected CLI binaries.

        Returns:
            Dictionary of binary name to detection result
        """
        # Extract all cached results from TTLCache
        results: dict[str, CLIDetectionResult] = {}
        if hasattr(self._detection_cache, "_cache"):
            for key, (result, _) in self._detection_cache._cache.items():
                if isinstance(key, str) and isinstance(result, CLIDetectionResult):
                    results[key] = result
        return results

    async def detect_multiple(
        self,
        binaries: list[tuple[str, str | None]],
        parallel: bool = True,
    ) -> dict[str, CLIDetectionResult]:
        """Detect multiple CLI binaries.

        Args:
            binaries: List of (binary_name, package_name) tuples
            parallel: Whether to detect in parallel

        Returns:
            Dictionary of binary name to detection result
        """
        if parallel:
            # Detect in parallel
            tasks = [
                self.detect_cli(binary_name, package_name)
                for binary_name, package_name in binaries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            detected: dict[str, CLIDetectionResult] = {}
            for (binary_name, _), result in zip(binaries, results, strict=False):
                if isinstance(result, Exception):
                    logger.error(
                        "cli_detection_error",
                        binary=binary_name,
                        error=str(result),
                    )
                elif isinstance(result, CLIDetectionResult):
                    detected[binary_name] = result

            return detected
        else:
            # Detect sequentially
            detected = {}
            for binary_name, package_name in binaries:
                try:
                    result = await self.detect_cli(binary_name, package_name)
                    detected[binary_name] = result
                except Exception as e:
                    logger.error(
                        "cli_detection_error",
                        binary=binary_name,
                        error=str(e),
                    )

            return detected
