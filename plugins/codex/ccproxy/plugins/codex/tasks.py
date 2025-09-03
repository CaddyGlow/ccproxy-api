"""Scheduled tasks for Codex plugin."""

from typing import TYPE_CHECKING, Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.scheduler.tasks import BaseScheduledTask


if TYPE_CHECKING:
    from .detection_service import CodexDetectionService


logger = get_plugin_logger()


class CodexDetectionRefreshTask(BaseScheduledTask):
    """Task to periodically refresh Codex CLI detection headers."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        detection_service: "CodexDetectionService",
        enabled: bool = True,
        skip_initial_run: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the Codex detection refresh task.

        Args:
            name: Task name
            interval_seconds: Interval between refreshes
            detection_service: The Codex detection service to refresh
            enabled: Whether the task is enabled
            skip_initial_run: Whether to skip the initial run at startup
            **kwargs: Additional arguments for BaseScheduledTask
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
        """Execute the detection refresh.

        Returns:
            True if refresh was successful, False otherwise
        """
        # Skip the first run if configured to do so
        if self._first_run and self.skip_initial_run:
            self._first_run = False
            logger.debug(
                "codex_detection_refresh_skipped_initial",
                task_name=self.name,
                reason="Initial run skipped to avoid duplicate detection at startup",
            )
            return True  # Return success to avoid triggering backoff

        self._first_run = False

        try:
            logger.info(
                "codex_detection_refresh_starting",
                task_name=self.name,
            )

            # Refresh the detection data
            detection_data = await self.detection_service.initialize_detection()

            logger.info(
                "codex_detection_refresh_completed",
                task_name=self.name,
                version=detection_data.codex_version if detection_data else "unknown",
                has_cached_data=detection_data is not None,
            )

            return True

        except Exception as e:
            logger.error(
                "codex_detection_refresh_failed",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def setup(self) -> None:
        """Perform any setup required before task execution starts."""
        logger.debug(
            "codex_detection_refresh_setup",
            task_name=self.name,
            interval_seconds=self.interval_seconds,
        )

    async def cleanup(self) -> None:
        """Perform any cleanup required after task execution stops."""
        logger.info(
            "codex_detection_refresh_cleanup",
            task_name=self.name,
        )
