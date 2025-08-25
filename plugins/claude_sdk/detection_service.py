"""Claude SDK CLI detection service using centralized detection."""

from typing import NamedTuple

from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.cli_detection import CLIDetectionService
from ccproxy.utils.caching import async_ttl_cache


logger = get_plugin_logger()


class ClaudeDetectionData(NamedTuple):
    """Detection data for Claude CLI."""

    claude_version: str | None
    cli_command: list[str] | None
    is_available: bool


class ClaudeSDKDetectionService:
    """Service for detecting Claude CLI availability.

    This detection service checks if the Claude CLI exists either as a direct
    binary in PATH or via package manager execution (e.g., bunx). Unlike the
    Claude API plugin, this doesn't support fallback mode as the SDK requires
    the actual CLI to be present.
    """

    def __init__(
        self, settings: Settings, cli_service: CLIDetectionService | None = None
    ) -> None:
        """Initialize the Claude SDK detection service.

        Args:
            settings: Application settings
            cli_service: Optional CLI detection service instance. If None, creates a new one.
        """
        self.settings = settings
        self._cli_service = cli_service or CLIDetectionService(settings)
        self._version: str | None = None
        self._cli_command: list[str] | None = None
        self._is_available = False

    @async_ttl_cache(maxsize=16, ttl=600.0)  # 10 minute cache for CLI detection
    async def initialize_detection(self) -> ClaudeDetectionData:
        """Initialize Claude CLI detection with caching.

        Returns:
            ClaudeDetectionData with detection results

        Note:
            No fallback support - SDK requires actual CLI presence
        """
        logger.debug("detection_starting", category="plugin")

        # Use centralized CLI detection service
        # For SDK, we don't want fallback - require actual CLI
        original_fallback = self._cli_service.resolver.fallback_enabled
        self._cli_service.resolver.fallback_enabled = False

        try:
            result = await self._cli_service.detect_cli(
                binary_name="claude",
                package_name="@anthropic-ai/claude-code",
                version_flag="--version",
                fallback_data=None,  # No fallback for SDK
                cache_key="claude_sdk",
            )

            # Accept both direct binary and package manager execution
            if result.is_available:
                self._version = result.version
                self._cli_command = result.command
                self._is_available = True
                logger.debug(
                    "cli_detection_completed",
                    cli_command=self._cli_command,
                    version=self._version,
                    source=result.source,
                    cached=hasattr(result, "cached") and result.cached,
                    category="plugin",
                )
            else:
                self._is_available = False
                logger.error(
                    "claude_sdk_detection_failed",
                    message="Claude CLI not found - SDK plugin cannot function without CLI",
                    category="plugin",
                )
        finally:
            # Restore original fallback setting
            self._cli_service.resolver.fallback_enabled = original_fallback

        return ClaudeDetectionData(
            claude_version=self._version,
            cli_command=self._cli_command,
            is_available=self._is_available,
        )

    def get_version(self) -> str | None:
        """Get the detected Claude CLI version.

        Returns:
            Version string if available, None otherwise
        """
        return self._version

    def get_cli_path(self) -> list[str] | None:
        """Get the detected Claude CLI command.

        Returns:
            CLI command list if available, None otherwise
        """
        return self._cli_command

    def is_claude_available(self) -> bool:
        """Check if Claude CLI is available.

        Returns:
            True if Claude CLI was detected, False otherwise
        """
        return self._is_available

    def invalidate_cache(self) -> None:
        """Clear all cached detection data."""
        # Clear the async cache for initialize_detection
        if hasattr(self.initialize_detection, "cache_clear"):
            self.initialize_detection.cache_clear()
        logger.debug("detection_cache_cleared", category="plugin")
