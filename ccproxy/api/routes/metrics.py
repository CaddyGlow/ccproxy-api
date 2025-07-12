"""Metrics endpoints for Claude Code Proxy API Server."""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from ccproxy.api.dependencies import MetricsCollectorDep, MetricsServiceDep
from ccproxy.metrics.exporters.json_api import JsonApiExporter
from ccproxy.metrics.exporters.sse import SSEMetricsExporter
from ccproxy.metrics.models import MetricType
from ccproxy.metrics.storage.memory import MemoryMetricsStorage


# Create the router for metrics endpoints
router = APIRouter(prefix="/metrics", tags=["metrics"])


# Global SSE exporter instance (will be initialized on first use)
_sse_exporter: SSEMetricsExporter | None = None


async def get_sse_exporter() -> SSEMetricsExporter:
    """Get or create the global SSE exporter instance."""
    global _sse_exporter
    if _sse_exporter is None:
        # Use memory storage for now (could be configurable)
        storage = MemoryMetricsStorage()
        _sse_exporter = SSEMetricsExporter(storage=storage)
        await _sse_exporter.start()
    return _sse_exporter


@router.get("/status")
async def metrics_status() -> dict[str, str]:
    """Get metrics status."""
    return {"status": "metrics endpoint available"}


@router.get("/data")
async def get_metrics_data(
    metrics_service: MetricsServiceDep,
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
        # Get the storage from metrics service
        storage = metrics_service.collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get metrics data
        return await json_exporter.get_metrics(
            start_time=start_time,
            end_time=end_time,
            metric_type=metric_type,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve metrics: {e}")


@router.get("/summary")
async def get_metrics_summary(
    metrics_service: MetricsServiceDep,
    start_time: datetime | None = Query(None, description="Start time for summary"),
    end_time: datetime | None = Query(None, description="End time for summary"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    session_id: str | None = Query(None, description="Filter by session ID"),
) -> dict[str, Any]:
    """Get aggregated metrics summary."""
    try:
        # Get the storage from metrics service
        storage = metrics_service.collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get summary data
        return await json_exporter.get_summary(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve summary: {e}")


@router.get("/stream")
async def stream_metrics(
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
        sse_exporter = await get_sse_exporter()

        async def event_generator():
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
        raise HTTPException(status_code=500, detail=f"Failed to start SSE stream: {e}")


@router.get("/connections")
async def get_sse_connections_info() -> dict[str, Any]:
    """Get information about active SSE connections."""
    try:
        sse_exporter = await get_sse_exporter()
        return await sse_exporter.get_connections_info()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get connections info: {e}"
        )


@router.get("/health")
async def get_metrics_health(
    metrics_service: MetricsServiceDep,
) -> dict[str, Any]:
    """Get health status of the metrics system."""
    try:
        # Get the storage from metrics service
        storage = metrics_service.collector.storage

        # Create JSON API exporter
        json_exporter = JsonApiExporter(storage)

        # Get health data
        health_data = await json_exporter.get_health()

        # Add SSE exporter health if available
        global _sse_exporter
        if _sse_exporter:
            sse_health = await _sse_exporter.health_check()
            connections_info = await _sse_exporter.get_connections_info()
            health_data["sse"] = {
                "healthy": sse_health,
                "connections": connections_info["total_connections"],
                "max_connections": connections_info["max_connections"],
            }
        else:
            health_data["sse"] = {"healthy": False, "reason": "not_started"}

        return health_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get health status: {e}")
