"""Metrics endpoints for Claude Code Proxy API Server."""

import time
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlmodel import Session, func, select

from ccproxy.api.dependencies import (
    ObservabilityMetricsDep,
)
from ccproxy.observability.storage.models import AccessLog


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])


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
        raise HTTPException(
            status_code=500, detail=f"Failed to generate Prometheus metrics: {str(e)}"
        ) from e


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


@router.get("/query")
async def query_metrics(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results"),
    start_time: float | None = Query(None, description="Start timestamp filter"),
    end_time: float | None = Query(None, description="End timestamp filter"),
    model: str | None = Query(None, description="Model filter"),
    service_type: str | None = Query(None, description="Service type filter"),
) -> dict[str, Any]:
    """
    Query access logs with filters.

    Returns access log entries with optional filtering by time range, model, and service type.
    """
    try:
        storage = await get_storage_backend()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage backend not available. Ensure DuckDB is installed and pipeline is running.",
            )

        # Use SQLModel for querying
        if hasattr(storage, "_engine") and storage._engine:
            try:
                with Session(storage._engine) as session:
                    # Build base query
                    statement = select(AccessLog)

                    # Add filters - convert Unix timestamps to datetime
                    if start_time:
                        start_dt = datetime.fromtimestamp(start_time)
                        statement = statement.where(AccessLog.timestamp >= start_dt)
                    if end_time:
                        end_dt = datetime.fromtimestamp(end_time)
                        statement = statement.where(AccessLog.timestamp <= end_dt)
                    if model:
                        statement = statement.where(AccessLog.model == model)
                    if service_type:
                        statement = statement.where(
                            AccessLog.service_type == service_type
                        )

                    # Apply limit and order
                    statement = statement.order_by(AccessLog.timestamp.desc()).limit(
                        limit
                    )

                    # Execute query
                    results = session.exec(statement).all()

                    # Convert to dict format
                    entries = [log.dict() for log in results]

                    return {
                        "results": entries,
                        "count": len(entries),
                        "limit": limit,
                        "filters": {
                            "start_time": start_time,
                            "end_time": end_time,
                            "model": model,
                            "service_type": service_type,
                        },
                        "timestamp": time.time(),
                    }

            except Exception as e:
                import structlog

                logger = structlog.get_logger(__name__)
                logger.error("sqlmodel_query_error", error=str(e))
                raise HTTPException(
                    status_code=500, detail=f"Query execution failed: {str(e)}"
                ) from e
        else:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Query execution failed: {str(e)}"
        ) from e


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

        # Use SQLModel for analytics
        if hasattr(storage, "_engine") and storage._engine:
            try:
                with Session(storage._engine) as session:
                    # Build base query
                    statement = select(AccessLog)

                    # Add filters - convert Unix timestamps to datetime
                    if start_time:
                        start_dt = datetime.fromtimestamp(start_time)
                        statement = statement.where(AccessLog.timestamp >= start_dt)
                    if end_time:
                        end_dt = datetime.fromtimestamp(end_time)
                        statement = statement.where(AccessLog.timestamp <= end_dt)
                    if model:
                        statement = statement.where(AccessLog.model == model)
                    if service_type:
                        statement = statement.where(
                            AccessLog.service_type == service_type
                        )

                    # Get summary statistics
                    summary_statement = select(
                        func.count().label("total_requests"),
                        func.avg(AccessLog.duration_ms).label("avg_duration_ms"),
                        func.sum(AccessLog.cost_usd).label("total_cost_usd"),
                        func.sum(AccessLog.tokens_input).label("total_tokens_input"),
                        func.sum(AccessLog.tokens_output).label("total_tokens_output"),
                    )

                    # Apply same filters to summary
                    if start_time:
                        start_dt = datetime.fromtimestamp(start_time)
                        summary_statement = summary_statement.where(
                            AccessLog.timestamp >= start_dt
                        )
                    if end_time:
                        end_dt = datetime.fromtimestamp(end_time)
                        summary_statement = summary_statement.where(
                            AccessLog.timestamp <= end_dt
                        )
                    if model:
                        summary_statement = summary_statement.where(
                            AccessLog.model == model
                        )
                    if service_type:
                        summary_statement = summary_statement.where(
                            AccessLog.service_type == service_type
                        )

                    summary_result = session.exec(summary_statement).first()

                    analytics = {
                        "summary": {
                            "total_requests": summary_result.total_requests
                            if summary_result
                            else 0,
                            "avg_duration_ms": summary_result.avg_duration_ms
                            if summary_result
                            else 0,
                            "total_cost_usd": summary_result.total_cost_usd
                            if summary_result
                            else 0,
                            "total_tokens_input": summary_result.total_tokens_input
                            if summary_result
                            else 0,
                            "total_tokens_output": summary_result.total_tokens_output
                            if summary_result
                            else 0,
                        },
                        "query_time": time.time(),
                        "backend": "sqlmodel",
                    }

                    # Add metadata
                    analytics["query_params"] = {
                        "start_time": start_time,
                        "end_time": end_time,
                        "model": model,
                        "service_type": service_type,
                        "hours": hours,
                    }

                    return analytics

            except Exception as e:
                import structlog

                logger = structlog.get_logger(__name__)
                logger.error("sqlmodel_analytics_error", error=str(e))
                raise HTTPException(
                    status_code=500, detail=f"Analytics query failed: {str(e)}"
                ) from e
        else:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analytics generation failed: {str(e)}"
        ) from e


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


@router.get("/entries")
async def get_database_entries(
    limit: int = Query(
        50, ge=1, le=1000, description="Maximum number of entries to return"
    ),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
    order_by: str = Query(
        "timestamp",
        description="Column to order by (timestamp, duration_ms, cost_usd, model, service_type, status_code)",
    ),
    order_desc: bool = Query(False, description="Order in descending order"),
) -> dict[str, Any]:
    """
    Get the last n database entries from the access logs.

    Returns individual request entries with full details for analysis.
    """
    try:
        storage = await get_storage_backend()
        if not storage:
            raise HTTPException(
                status_code=503,
                detail="Storage backend not available. Ensure DuckDB is installed and pipeline is running.",
            )

        # Use SQLModel for entries
        if hasattr(storage, "_engine") and storage._engine:
            try:
                with Session(storage._engine) as session:
                    # Validate order_by parameter using SQLModel
                    valid_columns = list(AccessLog.model_fields.keys())
                    if order_by not in valid_columns:
                        order_by = "timestamp"

                    # Build SQLModel query
                    order_attr = getattr(AccessLog, order_by)
                    order_clause = order_attr.desc() if order_desc else order_attr.asc()

                    statement = (
                        select(AccessLog)
                        .order_by(order_clause)
                        .offset(offset)
                        .limit(limit)
                    )
                    results = session.exec(statement).all()

                    # Get total count
                    count_statement = select(func.count()).select_from(AccessLog)
                    total_count = session.exec(count_statement).first()

                    # Convert to dict format
                    entries = [log.dict() for log in results]

                    return {
                        "entries": entries,
                        "total_count": total_count,
                        "limit": limit,
                        "offset": offset,
                        "order_by": order_by,
                        "order_desc": order_desc,
                        "page": (offset // limit) + 1,
                        "total_pages": ((total_count or 0) + limit - 1) // limit,
                        "backend": "sqlmodel",
                    }

            except Exception as e:
                import structlog

                logger = structlog.get_logger(__name__)
                logger.error("sqlmodel_entries_error", error=str(e))
                raise HTTPException(
                    status_code=500, detail=f"Failed to retrieve entries: {str(e)}"
                ) from e
        else:
            raise HTTPException(
                status_code=503,
                detail="Storage engine not available",
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve database entries: {str(e)}"
        ) from e


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
