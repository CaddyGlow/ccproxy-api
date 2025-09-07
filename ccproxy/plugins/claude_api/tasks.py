"""Scheduled tasks for Claude API plugin."""

from typing import TYPE_CHECKING, Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.scheduler.tasks import BaseScheduledTask


if TYPE_CHECKING:
    from .detection_service import ClaudeAPIDetectionService


logger = get_plugin_logger()


class ClaudeAPIDetectionRefreshTask(BaseScheduledTask):
    """Task to periodically refresh Claude CLI detection headers."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        detection_service: "ClaudeAPIDetectionService",
        enabled: bool = True,
        skip_initial_run: bool = True,
        **kwargs: Any,
    ):
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
        """Execute the detection refresh."""
        if self._first_run and self.skip_initial_run:
            self._first_run = False
            logger.debug(
                "claude_api_detection_refresh_skipped_initial",
                task_name=self.name,
            )
            return True

        self._first_run = False

        try:
            logger.info(
                "claude_api_detection_refresh_starting",
                task_name=self.name,
            )
            detection_data = await self.detection_service.initialize_detection()

            logger.info(
                "claude_api_detection_refresh_completed",
                task_name=self.name,
                version=detection_data.claude_version if detection_data else "unknown",
            )
            return True

        except Exception as e:
            logger.error(
                "claude_api_detection_refresh_failed",
                task_name=self.name,
                error=str(e),
            )
            return False

    async def setup(self) -> None:
        """Setup before task execution starts."""
        logger.debug(
            "claude_api_detection_refresh_setup",
            task_name=self.name,
        )

    async def cleanup(self) -> None:
        """Cleanup after task execution stops."""
        logger.info(
            "claude_api_detection_refresh_cleanup",
            task_name=self.name,
        )
