"""Prometheus Pushgateway integration for batch metrics."""

from __future__ import annotations

import logging
from typing import Any

from ccproxy.config.observability import ObservabilitySettings


logger = logging.getLogger(__name__)


# Import prometheus_client with graceful degradation (matching existing metrics.py pattern)
try:
    from prometheus_client import CollectorRegistry, push_to_gateway

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Dummy classes for graceful degradation
    def push_to_gateway(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    class CollectorRegistry:  # type: ignore[no-redef]
        pass


class PushgatewayClient:
    """Simple Prometheus Pushgateway client."""

    def __init__(self, settings: ObservabilitySettings) -> None:
        """Initialize Pushgateway client.

        Args:
            settings: Observability configuration settings
        """
        self.settings = settings
        self._enabled = PROMETHEUS_AVAILABLE and settings.pushgateway_enabled

        if not PROMETHEUS_AVAILABLE and settings.pushgateway_enabled:
            logger.warning(
                "prometheus_client not available. Pushgateway will be disabled. "
                "Install with: pip install prometheus-client"
            )

    def push_metrics(self, registry: CollectorRegistry) -> bool:
        """Push metrics to Pushgateway.

        Args:
            registry: Prometheus metrics registry to push

        Returns:
            True if push succeeded, False otherwise
        """
        if not self._enabled or not self.settings.pushgateway_url:
            return False

        try:
            push_to_gateway(
                gateway=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
                registry=registry,
            )
            logger.debug(
                "Pushgateway push success: url=%s job=%s",
                self.settings.pushgateway_url,
                self.settings.pushgateway_job,
            )
            return True

        except Exception as e:
            logger.error(
                "Pushgateway push failed: url=%s job=%s error=%s",
                self.settings.pushgateway_url,
                self.settings.pushgateway_job,
                str(e),
            )
            return False

    def is_enabled(self) -> bool:
        """Check if Pushgateway client is enabled and configured."""
        return self._enabled and bool(self.settings.pushgateway_url)
