"""Prometheus Pushgateway integration for batch metrics."""

from __future__ import annotations

from typing import Any

from ccproxy.config.observability import ObservabilitySettings
from ccproxy.core.logging import get_structlog_logger


logger = get_structlog_logger(__name__)


# Import prometheus_client with graceful degradation (matching existing metrics.py pattern)
try:
    from prometheus_client import (
        CollectorRegistry,
        delete_from_gateway,
        push_to_gateway,
        pushadd_to_gateway,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Dummy classes for graceful degradation
    def push_to_gateway(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def pushadd_to_gateway(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    def delete_from_gateway(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        pass

    class CollectorRegistry:  # type: ignore[no-redef]
        pass


class PushgatewayClient:
    """Prometheus Pushgateway client using official prometheus_client methods.

    Supports standard pushgateway operations:
    - push_to_gateway(): Replace all metrics for job/instance
    - pushadd_to_gateway(): Add metrics to existing job/instance
    - delete_from_gateway(): Delete metrics for job/instance

    Also supports VictoriaMetrics remote write protocol for compatibility.
    """

    def __init__(self, settings: ObservabilitySettings) -> None:
        """Initialize Pushgateway client.

        Args:
            settings: Observability configuration settings
        """
        self.settings = settings
        self._enabled = PROMETHEUS_AVAILABLE and settings.pushgateway_enabled

        # Only log if pushgateway is enabled but prometheus is not available
        if settings.pushgateway_enabled and not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not available. Pushgateway will be disabled. "
                "Install with: pip install prometheus-client"
            )

    def push_metrics(self, registry: CollectorRegistry, method: str = "push") -> bool:
        """Push metrics to Pushgateway using official prometheus_client methods.

        Args:
            registry: Prometheus metrics registry to push
            method: Push method - "push" (replace), "pushadd" (add), or "delete"

        Returns:
            True if push succeeded, False otherwise
        """

        if not self._enabled or not self.settings.pushgateway_url:
            return False

        try:
            # Check if URL looks like VictoriaMetrics remote write endpoint
            if "/api/v1/write" in self.settings.pushgateway_url:
                return self._push_remote_write(registry)
            else:
                return self._push_standard(registry, method)

        except Exception as e:
            logger.error(
                "pushgateway_push_failed",
                url=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
                method=method,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def _push_standard(self, registry: CollectorRegistry, method: str = "push") -> bool:
        """Push using standard Prometheus pushgateway protocol with official client methods.

        Args:
            registry: Prometheus metrics registry
            method: Push method - "push" (replace), "pushadd" (add), or "delete"
        """
        if not self.settings.pushgateway_url:
            return False

        # Use the appropriate prometheus_client function based on method
        if method == "push":
            push_to_gateway(
                gateway=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
                registry=registry,
            )
        elif method == "pushadd":
            pushadd_to_gateway(
                gateway=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
                registry=registry,
            )
        elif method == "delete":
            delete_from_gateway(
                gateway=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
            )
        else:
            logger.error("pushgateway_invalid_method", method=method)
            return False

        logger.debug(
            "pushgateway_push_success",
            url=self.settings.pushgateway_url,
            job=self.settings.pushgateway_job,
            protocol="standard",
            method=method,
        )
        return True

    def _push_remote_write(self, registry: CollectorRegistry) -> bool:
        """Push using VictoriaMetrics import protocol for exposition format data.

        VictoriaMetrics supports importing Prometheus exposition format data
        via the /api/v1/import/prometheus endpoint, which is simpler than
        the full remote write protocol that requires protobuf encoding.
        """
        import requests  # type: ignore[import-untyped]
        from prometheus_client.exposition import generate_latest

        if not self.settings.pushgateway_url:
            return False

        # Generate metrics in Prometheus exposition format
        metrics_data = generate_latest(registry)

        # Convert /api/v1/write URL to /api/v1/import/prometheus for VictoriaMetrics
        # This endpoint accepts Prometheus exposition format directly
        if "/api/v1/write" in self.settings.pushgateway_url:
            import_url = self.settings.pushgateway_url.replace(
                "/api/v1/write", "/api/v1/import/prometheus"
            )
        else:
            # Fallback - assume it's already the correct import URL
            import_url = self.settings.pushgateway_url

        # VictoriaMetrics import endpoint accepts text/plain exposition format
        response = requests.post(
            import_url,
            data=metrics_data,
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "User-Agent": "ccproxy-pushgateway-client/1.0",
            },
            timeout=30,
        )

        if response.status_code in (200, 204):
            logger.debug(
                "pushgateway_import_success",
                url=import_url,
                job=self.settings.pushgateway_job,
                protocol="victoriametrics_import",
                status=response.status_code,
            )
            return True
        else:
            logger.error(
                "pushgateway_import_failed",
                url=import_url,
                status=response.status_code,
                response=response.text[:500] if response.text else "empty",
            )
            return False

    def push_add_metrics(self, registry: CollectorRegistry) -> bool:
        """Add metrics to existing job/instance (pushadd operation).

        Args:
            registry: Prometheus metrics registry to add

        Returns:
            True if push succeeded, False otherwise
        """
        return self.push_metrics(registry, method="pushadd")

    def delete_metrics(self) -> bool:
        """Delete all metrics for the configured job.

        Returns:
            True if delete succeeded, False otherwise
        """

        if not self._enabled or not self.settings.pushgateway_url:
            return False

        try:
            # Only standard pushgateway supports delete operation
            if "/api/v1/write" in self.settings.pushgateway_url:
                logger.warning("pushgateway_delete_not_supported_for_remote_write")
                return False
            else:
                return self._push_standard(None, method="delete")  # type: ignore[arg-type]
        except Exception as e:
            logger.error(
                "pushgateway_delete_failed",
                url=self.settings.pushgateway_url,
                job=self.settings.pushgateway_job,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def is_enabled(self) -> bool:
        """Check if Pushgateway client is enabled and configured."""
        return self._enabled and bool(self.settings.pushgateway_url)


# Global pushgateway client instance
_global_pushgateway_client: PushgatewayClient | None = None


def get_pushgateway_client() -> PushgatewayClient:
    """Get or create global pushgateway client instance."""
    global _global_pushgateway_client

    if _global_pushgateway_client is None:
        # Import here to avoid circular imports
        from ccproxy.config.settings import get_settings

        settings = get_settings()
        _global_pushgateway_client = PushgatewayClient(settings.observability)

    return _global_pushgateway_client


def reset_pushgateway_client() -> None:
    """Reset global pushgateway client instance (mainly for testing)."""
    global _global_pushgateway_client
    _global_pushgateway_client = None
