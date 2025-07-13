"""Metrics endpoints for Claude Code Proxy API Server."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from ccproxy.api.dependencies import (
    ObservabilityMetricsDep,
)


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/status")
async def metrics_status(metrics: ObservabilityMetricsDep) -> dict[str, str]:
    """Get observability system status."""
    return {
        "status": "healthy",
        "prometheus_enabled": metrics.is_enabled(),
        "observability_system": "hybrid_prometheus_structlog",
    }


@router.get("/dashboard")
async def get_metrics_dashboard() -> HTMLResponse:
    """Serve the metrics dashboard SPA entry point."""
    from pathlib import Path

    # Get the path to the dashboard folder
    current_file = Path(__file__)
    project_root = (
        current_file.parent.parent.parent.parent
    )  # ccproxy/api/routes/metrics.py -> project root
    dashboard_folder = project_root / "ccproxy" / "static" / "dashboard"
    dashboard_index = dashboard_folder / "index.html"

    # Check if dashboard folder and index.html exist
    if not dashboard_folder.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard not found. Please build the dashboard first using 'cd dashboard && bun run build:prod'",
        )

    if not dashboard_index.exists():
        raise HTTPException(
            status_code=404,
            detail="Dashboard index.html not found. Please rebuild the dashboard using 'cd dashboard && bun run build:prod'",
        )

    # Read the HTML content
    try:
        with dashboard_index.open(encoding="utf-8") as f:
            html_content = f.read()

        return HTMLResponse(
            content=html_content,
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Content-Type": "text/html; charset=utf-8",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to serve dashboard: {str(e)}"
        ) from e


@router.get("/dashboard/favicon.svg")
async def get_dashboard_favicon() -> FileResponse:
    """Serve the dashboard favicon."""
    from pathlib import Path

    # Get the path to the favicon
    current_file = Path(__file__)
    project_root = (
        current_file.parent.parent.parent.parent
    )  # ccproxy/api/routes/metrics.py -> project root
    favicon_path = project_root / "ccproxy" / "static" / "dashboard" / "favicon.svg"

    if not favicon_path.exists():
        raise HTTPException(status_code=404, detail="Favicon not found")

    return FileResponse(
        path=str(favicon_path),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/prometheus")
async def get_prometheus_metrics(metrics: ObservabilityMetricsDep):
    """Export metrics in Prometheus format using native prometheus_client.

    This endpoint exposes operational metrics collected by the hybrid observability
    system for Prometheus scraping.

    Args:
        metrics: Observability metrics dependency

    Returns:
        Prometheus-formatted metrics text
    """
    try:
        # Check if prometheus_client is available
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        except ImportError as err:
            raise HTTPException(
                status_code=503,
                detail="Prometheus client not available. Install with: pip install prometheus-client",
            ) from err

        if not metrics.is_enabled():
            raise HTTPException(
                status_code=503,
                detail="Prometheus metrics not enabled. Ensure prometheus-client is installed.",
            )

        # Generate prometheus format using the registry
        prometheus_data = generate_latest(metrics.registry)

        # Return the metrics data with proper content type
        from fastapi import Response

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
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate Prometheus metrics: {str(e)}"
        ) from e
