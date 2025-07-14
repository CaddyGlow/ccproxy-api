"""Metrics endpoints for Claude Code Proxy API Server."""

import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from ccproxy.api.dependencies import (
    ObservabilityMetricsDep,
)
from ccproxy.core.logging import get_structlog_logger


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])

# Create structured logger
logger = get_structlog_logger(__name__)


@router.get("/status")
async def metrics_status(metrics: ObservabilityMetricsDep) -> dict[str, str]:
    """Get observability system status."""
    return {
        "status": "healthy",
        "prometheus_enabled": str(metrics.is_enabled()),
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
        logger.error(
            "Failed to serve dashboard",
            error=str(e),
            dashboard_path=str(dashboard_index),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Dashboard temporarily unavailable. Please check server logs for details.",
        ) from None


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
async def get_prometheus_metrics(metrics: ObservabilityMetricsDep) -> Any:
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
        from prometheus_client import REGISTRY, CollectorRegistry

        # Use the global registry if metrics.registry is None (default behavior)
        registry = metrics.registry if metrics.registry is not None else REGISTRY
        prometheus_data = generate_latest(registry)

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
        logger.error(
            "Failed to generate Prometheus metrics",
            error=str(e),
            error_type=type(e).__name__,
            metrics_enabled=metrics.is_enabled(),
        )
        raise HTTPException(
            status_code=500,
            detail="Metrics generation temporarily unavailable. Please check server logs for details.",
        ) from None


async def get_storage_backend() -> Any:
    """Get DuckDB storage backend from pipeline."""
    try:
        from ccproxy.observability.pipeline import get_pipeline

        pipeline = await get_pipeline()

        # Get DuckDB storage from pipeline backends
        for backend in pipeline._storage_backends:
            if hasattr(backend, "query"):  # DuckDB storage has query method
                return backend

        return None
    except Exception:
        return None


# SQL injection protection - allow only safe SQL patterns
SAFE_SQL_PATTERN = re.compile(
    r"^SELECT\s+.*\s+FROM\s+(?:requests|operations)(?:\s+WHERE\s+.*)?(?:\s+GROUP\s+BY\s+.*)?(?:\s+ORDER\s+BY\s+.*)?(?:\s+LIMIT\s+\d+)?;?$",
    re.IGNORECASE | re.DOTALL,
)


@router.get("/query")
async def query_metrics(
    sql: str = Query(..., description="SQL query to execute"),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
) -> dict[str, Any]:
    """
    Execute custom SQL query on metrics data.

    Supports queries on 'requests' and 'operations' tables with safety restrictions.

    Example queries:
    - SELECT COUNT(*) FROM requests WHERE timestamp > '2024-01-01'
    - SELECT model, AVG(response_time) FROM requests GROUP BY model
    - SELECT * FROM requests WHERE status = 'error' ORDER BY timestamp DESC
    """
    try:
        storage = await get_storage_backend()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage backend not available. Ensure DuckDB is installed and pipeline is running.",
            )

        # Basic SQL injection protection
        sql_clean = sql.strip()
        if not SAFE_SQL_PATTERN.match(sql_clean):
            raise HTTPException(
                status_code=400,
                detail="Invalid SQL query. Only SELECT queries on 'requests' and 'operations' tables are allowed.",
            )

        # Execute query
        results = await storage.query(sql_clean, limit=limit)

        return {
            "query": sql_clean,
            "results": results,
            "count": len(results),
            "limit": limit,
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Query execution failed",
            error=str(e),
            error_type=type(e).__name__,
            query=sql_clean if "sql_clean" in locals() else sql,
            limit=limit,
        )
        raise HTTPException(
            status_code=500,
            detail="Query execution failed. Please check query syntax and try again.",
        ) from None


@router.get("/analytics")
async def get_analytics(
    start_time: float | None = Query(None, description="Start timestamp (Unix time)"),
    end_time: float | None = Query(None, description="End timestamp (Unix time)"),
    model: str | None = Query(None, description="Filter by model name"),
    service_type: str | None = Query(
        None, description="Filter by service type (proxy_service or claude_sdk_service)"
    ),
    hours: int | None = Query(
        24, ge=1, le=168, description="Hours of data to analyze (default: 24)"
    ),
) -> dict[str, Any]:
    """
    Get comprehensive analytics for metrics data.

    Returns summary statistics, hourly trends, and model breakdowns.
    """
    try:
        storage = await get_storage_backend()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage backend not available. Ensure DuckDB is installed and pipeline is running.",
            )

        # Default time range if not provided
        if start_time is None and end_time is None and hours:
            end_time = time.time()
            start_time = end_time - (hours * 3600)

        # Get analytics data
        analytics: dict[str, Any] = await storage.get_analytics(
            start_time=start_time,
            end_time=end_time,
            model=model,
            service_type=service_type,
        )

        # Add metadata
        analytics["query_params"] = {
            "start_time": start_time,
            "end_time": end_time,
            "model": model,
            "service_type": service_type,
            "hours": hours,
        }

        return analytics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Analytics generation failed",
            error=str(e),
            error_type=type(e).__name__,
            start_time=start_time,
            end_time=end_time,
            model=model,
            service_type=service_type,
            hours=hours,
        )
        raise HTTPException(
            status_code=500,
            detail="Analytics temporarily unavailable. Please check server logs for details.",
        ) from None


@router.get("/stream")
async def stream_metrics() -> StreamingResponse:
    """
    Stream real-time metrics and request logs via Server-Sent Events.

    Returns a continuous stream of request events, metrics updates,
    and analytics data as they occur.
    """
    import asyncio
    import json
    from collections.abc import AsyncIterator

    def json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for datetime and other objects."""
        from datetime import datetime

        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def event_stream() -> AsyncIterator[str]:
        """Generate Server-Sent Events for real-time metrics."""

        # Send initial connection event
        data = json.dumps(
            {
                "type": "connection",
                "message": "Connected to metrics stream",
                "timestamp": time.time(),
            },
            default=json_serializer,
        )
        yield f"data: {data}\n\n"

        # Keep track of last request count to detect new requests
        last_request_count = 0

        try:
            while True:
                try:
                    # Get current analytics to detect new requests
                    storage = await get_storage_backend()
                    if storage:
                        # Get recent analytics (last 5 minutes)
                        current_time = time.time()
                        analytics = await storage.get_analytics(
                            start_time=current_time - 300,  # 5 minutes ago
                            end_time=current_time,
                        )

                        # Check if there are new requests
                        current_request_count = analytics.get("summary", {}).get(
                            "total_requests", 0
                        )

                        if current_request_count > last_request_count:
                            # New requests detected, send analytics update
                            data = json.dumps(
                                {
                                    "type": "analytics_update",
                                    "data": analytics,
                                    "timestamp": time.time(),
                                },
                                default=json_serializer,
                            )
                            yield f"data: {data}\n\n"

                            last_request_count = current_request_count

                        # Send periodic heartbeat with current stats
                        data = json.dumps(
                            {
                                "type": "heartbeat",
                                "stats": {
                                    "total_requests": current_request_count,
                                    "timestamp": time.time(),
                                },
                            },
                            default=json_serializer,
                        )
                        yield f"data: {data}\n\n"

                except Exception as e:
                    # Send error event but keep connection alive
                    data = json.dumps(
                        {
                            "type": "error",
                            "message": str(e),
                            "timestamp": time.time(),
                        },
                        default=json_serializer,
                    )
                    yield f"data: {data}\n\n"

                # Wait before next check
                await asyncio.sleep(2)  # Check every 2 seconds

        except asyncio.CancelledError:
            # Send disconnect event
            data = json.dumps(
                {
                    "type": "disconnect",
                    "message": "Stream disconnected",
                    "timestamp": time.time(),
                },
                default=json_serializer,
            )
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )


@router.get("/health")
async def get_storage_health() -> dict[str, Any]:
    """Get health status of the storage backend."""
    try:
        storage = await get_storage_backend()
        if not storage:
            return {
                "status": "unavailable",
                "storage_backend": "none",
                "message": "No storage backend available",
            }

        health: dict[str, Any] = await storage.health_check()
        health["storage_backend"] = "duckdb"

        return health

    except Exception as e:
        return {"status": "error", "storage_backend": "duckdb", "error": str(e)}
