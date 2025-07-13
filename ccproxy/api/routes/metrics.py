"""Metrics endpoints for Claude Code Proxy API Server."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ccproxy.api.dependencies import MetricsCollectorDep, MetricsServiceDep
from ccproxy.metrics.exporters.json_api import JsonApiExporter
from ccproxy.metrics.exporters.sse import SSEMetricsExporter
from ccproxy.metrics.models import MetricType
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage


def ensure_timezone_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware (convert naive to UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=UTC)
    return dt


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])


async def get_sse_exporter(collector: MetricsCollectorDep) -> SSEMetricsExporter:
    """Get or create the SSE exporter instance from the metrics collector."""
    # Check if the collector already has an SSE exporter
    if collector.sse_exporter:
        return collector.sse_exporter

    # Create a new SSE exporter and attach it to the collector
    sse_exporter = SSEMetricsExporter(storage=collector.storage)
    await sse_exporter.start()

    # Set it on the collector for future use
    collector.sse_exporter = sse_exporter
    return sse_exporter


@router.get("/status")
async def metrics_status() -> dict[str, str]:
    """Get metrics status."""
    return {"status": "metrics endpoint available"}


@router.get("/data")
async def get_metrics_data(
    collector: MetricsCollectorDep,
    start_time: datetime | None = Query(
        None, description="Start time for metrics query"
    ),
    end_time: datetime | None = Query(None, description="End time for metrics query"),
    metric_type: MetricType | None = Query(None, description="Filter by metric type"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
) -> dict[str, Any]:
    """Get metrics data with filtering and pagination."""
    try:
        # Get the storage from metrics collector
        storage = collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get metrics data
        return await json_exporter.get_metrics(
            start_time=ensure_timezone_aware(start_time),
            end_time=ensure_timezone_aware(end_time),
            metric_type=metric_type,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve metrics: {e}"
        ) from e


@router.get("/summary")
async def get_metrics_summary(
    collector: MetricsCollectorDep,
    start_time: datetime | None = Query(None, description="Start time for summary"),
    end_time: datetime | None = Query(None, description="End time for summary"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
) -> dict[str, Any]:
    """Get aggregated metrics summary."""
    try:
        # Get the storage from metrics collector
        storage = collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get summary data
        return await json_exporter.get_summary(
            start_time=ensure_timezone_aware(start_time),
            end_time=ensure_timezone_aware(end_time),
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve summary: {e}"
        ) from e


@router.get("/stream")
async def stream_metrics(
    collector: MetricsCollectorDep,
    metric_types: list[MetricType] | None = Query(
        None, description="Metric types to subscribe to"
    ),
    user_id: str | None = Query(None, description="Filter by user ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
    subscription_types: list[str] | None = Query(
        ["live"], description="Subscription types: live, summary, time_series"
    ),
) -> StreamingResponse:
    """Stream real-time metrics via Server-Sent Events (SSE)."""
    try:
        sse_exporter = await get_sse_exporter(collector)

        async def event_generator() -> AsyncIterator[str]:
            """Generate SSE events for the client."""
            async with sse_exporter.create_connection(
                metric_types=metric_types,
                user_id=user_id,
                session_id=session_id,
                subscription_types=subscription_types,
            ) as (connection_id, event_stream):
                async for event in event_stream:
                    yield event

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start SSE stream: {e}"
        ) from e


@router.get("/connections")
async def get_sse_connections_info(
    collector: MetricsCollectorDep,
) -> dict[str, Any]:
    """Get information about active SSE connections."""
    try:
        sse_exporter = await get_sse_exporter(collector)
        return await sse_exporter.get_connections_info()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get connections info: {e}"
        ) from e


@router.get("/health")
async def get_metrics_health(
    collector: MetricsCollectorDep,
) -> dict[str, Any]:
    """Get health status of the metrics system."""
    try:
        # Get the storage from metrics collector
        storage = collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get health data
        health_data = await json_exporter.get_health()

        # Add SSE exporter health if available
        if collector.sse_exporter:
            sse_exporter = collector.sse_exporter
            sse_health = await sse_exporter.health_check()
            connections_info = await sse_exporter.get_connections_info()
            health_data["sse"] = {
                "healthy": sse_health,
                "connections": connections_info["total_connections"],
                "max_connections": connections_info["max_connections"],
            }
        else:
            health_data["sse"] = {"healthy": False, "reason": "not_started"}

        return health_data
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get health status: {e}"
        ) from e


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
