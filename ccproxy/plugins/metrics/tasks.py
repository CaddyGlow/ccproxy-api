"""Scheduled tasks for the metrics plugin."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.scheduler.tasks import BaseScheduledTask

from .pushgateway import PushgatewayClient


logger = get_plugin_logger(__name__)


class PushgatewayTask(BaseScheduledTask):
    """Task for pushing metrics to Pushgateway periodically."""

    def __init__(
        self,
        name: str,
        interval_seconds: float,
        enabled: bool = True,
        max_backoff_seconds: float = 300.0,
        metrics_config: Any | None = None,
        metrics_hook: Any | None = None,
    ):
        """
        Initialize pushgateway task.

        Args:
            name: Task name
            interval_seconds: Interval between pushgateway operations
            enabled: Whether task is enabled
            max_backoff_seconds: Maximum backoff delay for failures
            metrics_config: Metrics plugin configuration
            metrics_hook: Metrics hook instance for getting collector
        """
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            enabled=enabled,
            max_backoff_seconds=max_backoff_seconds,
        )
        self._metrics_config = metrics_config
        self._metrics_hook = metrics_hook
        self._pushgateway_client: PushgatewayClient | None = None

    async def setup(self) -> None:
        """Initialize pushgateway client for operations."""
        try:
            if self._metrics_config and self._metrics_hook:
                self._pushgateway_client = PushgatewayClient(self._metrics_config)
                logger.debug(
                    "pushgateway_task_setup_complete",
                    task_name=self.name,
                    url=self._metrics_config.pushgateway_url,
                    job=self._metrics_config.pushgateway_job,
                )
            else:
                logger.warning(
                    "pushgateway_task_setup_missing_config",
                    task_name=self.name,
                    has_config=self._metrics_config is not None,
                    has_hook=self._metrics_hook is not None,
                )
        except Exception as e:
            logger.error(
                "pushgateway_task_setup_failed",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            raise

    async def run(self) -> bool:
        """Execute pushgateway metrics push."""
        try:
            if not self._pushgateway_client or not self._metrics_hook:
                logger.warning(
                    "pushgateway_no_client_or_hook",
                    task_name=self.name,
                    has_client=self._pushgateway_client is not None,
                    has_hook=self._metrics_hook is not None,
                )
                return False

            if not self._pushgateway_client.is_enabled():
                logger.debug("pushgateway_disabled", task_name=self.name)
                return True  # Not an error, just disabled

            # Get the metrics collector and push metrics
            collector = self._metrics_hook.get_collector()
            if not collector:
                logger.warning("pushgateway_no_collector", task_name=self.name)
                return False

            # Push metrics using the client
            success = self._pushgateway_client.push_metrics(
                collector.get_registry(), method="push"
            )

            if success:
                logger.debug("pushgateway_push_success", task_name=self.name)
            else:
                logger.warning("pushgateway_push_failed", task_name=self.name)

            return success

        except Exception as e:
            logger.error(
                "pushgateway_task_error",
                task_name=self.name,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=e,
            )
            return False
