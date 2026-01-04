"""Scheduled tasks for Claude SDK plugin."""

from typing import TYPE_CHECKING, Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.scheduler.tasks import BaseScheduledTask


if TYPE_CHECKING:
    from .detection_service import ClaudeSDKDetectionService


logger = get_plugin_logger()


class ClaudeSDKDetectionRefreshTask(BaseScheduledTask):
    """Task to periodically refresh Claude CLI detection."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        detection_service: "ClaudeSDKDetectionService",
        enabled: bool = True,
        skip_initial_run: bool = True,
        **kwargs: Any,
    ):
        """Initialize the detection refresh task.

        Args:
            name: Task name
            interval_seconds: How often to run the task
            detection_service: Claude CLI detection service
            enabled: Whether the task is enabled
            skip_initial_run: Whether to skip the first run
            **kwargs: Additional task arguments
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            enabled=enabled,
            **kwargs,
        )
        self.detection_service = detection_service
        self.skip_initial_run = skip_initial_run
        self._first_run = True

    async def run(self) -> bool:
        """Execute the Claude CLI detection refresh.

        Returns:
            True if successful, False otherwise
        """
        if self._first_run and self.skip_initial_run:
            self._first_run = False
            logger.debug(
                "claude_sdk_detection_refresh_skipped_initial",
                task_name=self.name,
            )
            return True

        self._first_run = False

        try:
            logger.debug(
                "claude_sdk_detection_refresh_starting",
                task_name=self.name,
            )

            # Refresh Claude CLI detection
            detection_data = await self.detection_service.initialize_detection()

            logger.debug(
                "claude_sdk_detection_refresh_completed",
                task_name=self.name,
                version=detection_data.claude_version or "unknown",
                cli_command=detection_data.cli_command,
                is_available=detection_data.is_available,
            )
            return True

        except Exception as e:
            logger.error(
                "claude_sdk_detection_refresh_failed",
                task_name=self.name,
                error=str(e),
                exc_info=e,
            )
            return False

    async def setup(self) -> None:
        """Setup before task execution starts."""
        pass

    async def cleanup(self) -> None:
        """Cleanup after task execution stops."""
        pass
