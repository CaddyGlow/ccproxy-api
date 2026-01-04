"""Metrics endpoints for the metrics plugin."""

from typing import Any

from fastapi import APIRouter, HTTPException, Response

from ccproxy.core.logging import get_plugin_logger

from .collector import PrometheusMetrics


logger = get_plugin_logger(__name__)


def create_metrics_router(collector: PrometheusMetrics | None) -> APIRouter:
    """Create metrics router with the given collector.

    Args:
        collector: Prometheus metrics collector instance

    Returns:
        FastAPI router with metrics endpoints
    """
    router = APIRouter(tags=["metrics"])

    @router.get("/metrics")
    async def get_prometheus_metrics() -> Response:
        """Export metrics in Prometheus format.

        This endpoint exposes operational metrics collected by the metrics plugin
        for Prometheus scraping.

        Returns:
            Prometheus-formatted metrics text
        """
        if not collector or not collector.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Metrics collection not enabled. Ensure prometheus-client is installed.",
            )

        try:
            # Check if prometheus_client is available
            try:
                from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            except ImportError as err:
                raise HTTPException(
                    status_code=503,
                    detail="Prometheus client not available. Install with: pip install prometheus-client",
                ) from err

            # Generate prometheus format using the registry
            from prometheus_client import REGISTRY

            # Use the collector's registry or fall back to global
            registry = (
                collector.registry if collector.registry is not None else REGISTRY
            )
            prometheus_data = generate_latest(registry)

            # Return the metrics data with proper content type
            return Response(
                content=prometheus_data,
                media_type=CONTENT_TYPE_LATEST,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )

        except HTTPException:
            raise
        except ImportError as e:
            logger.error(
                "prometheus_import_error",
                error=str(e),
                exc_info=e,
            )
            raise HTTPException(
                status_code=503, detail=f"Prometheus dependencies missing: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(
                "metrics_generation_error",
                error=str(e),
                exc_info=e,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate Prometheus metrics: {str(e)}",
            ) from e

    @router.get("/metrics/health")
    async def metrics_health() -> dict[str, Any]:
        """Get metrics system health status.

        Returns:
            Health status of the metrics collection system
        """
        return {
            "status": "healthy" if collector and collector.is_enabled() else "disabled",
            "prometheus_enabled": collector.is_enabled() if collector else False,
            "namespace": collector.namespace if collector else None,
        }

    return router
