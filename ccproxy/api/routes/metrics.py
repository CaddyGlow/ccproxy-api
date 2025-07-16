"""Metrics endpoints for Claude Code Proxy API Server."""

import time
from datetime import datetime as dt
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
                    start_dt = dt.fromtimestamp(start_time) if start_time else None
                    end_dt = dt.fromtimestamp(end_time) if end_time else None

                    if start_dt:
                        statement = statement.where(AccessLog.timestamp >= start_dt)
                    if end_dt:
                        statement = statement.where(AccessLog.timestamp <= end_dt)
                    if model:
                        statement = statement.where(AccessLog.model == model)
                    if service_type:
                        statement = statement.where(
                            AccessLog.service_type == service_type
                        )

                    # Apply limit and order
                    statement = statement.order_by(AccessLog.timestamp.desc()).limit(  # type: ignore[attr-defined]
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
        None,
        description="Filter by service type. Supports comma-separated values (e.g., 'proxy_service,sdk_service') and negation with ! prefix (e.g., '!access_log,!sdk_service')",
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
                    start_dt = dt.fromtimestamp(start_time) if start_time else None
                    end_dt = dt.fromtimestamp(end_time) if end_time else None

                    # Helper function to build filter conditions
                    def build_filter_conditions():
                        conditions = []
                        if start_dt:
                            conditions.append(AccessLog.timestamp >= start_dt)
                        if end_dt:
                            conditions.append(AccessLog.timestamp <= end_dt)
                        if model:
                            conditions.append(AccessLog.model == model)

                        # Apply service type filtering with comma-separated values and negation
                        if service_type:
                            service_filters = [
                                s.strip() for s in service_type.split(",")
                            ]
                            include_filters = [
                                f for f in service_filters if not f.startswith("!")
                            ]
                            exclude_filters = [
                                f[1:] for f in service_filters if f.startswith("!")
                            ]

                            if include_filters:
                                conditions.append(
                                    AccessLog.service_type.in_(include_filters)
                                )
                            if exclude_filters:
                                conditions.append(
                                    ~AccessLog.service_type.in_(exclude_filters)
                                )

                        return conditions

                    # Get summary statistics using individual queries to avoid overload issues
                    # Reuse datetime variables defined above

                    filter_conditions = build_filter_conditions()

                    total_requests = session.exec(
                        select(func.count())
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    avg_duration = session.exec(
                        select(func.avg(AccessLog.duration_ms))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    total_cost = session.exec(
                        select(func.sum(AccessLog.cost_usd))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    total_tokens_input = session.exec(
                        select(func.sum(AccessLog.tokens_input))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    total_tokens_output = session.exec(
                        select(func.sum(AccessLog.tokens_output))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    # Token analytics - all token types
                    total_cache_read_tokens = session.exec(
                        select(func.sum(AccessLog.cache_read_tokens))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    total_cache_write_tokens = session.exec(
                        select(func.sum(AccessLog.cache_write_tokens))
                        .select_from(AccessLog)
                        .where(*filter_conditions)
                    ).first()

                    # Success and error request analytics
                    success_conditions = filter_conditions + [
                        AccessLog.status_code >= 200,
                        AccessLog.status_code < 400,
                    ]
                    total_successful_requests = session.exec(
                        select(func.count())
                        .select_from(AccessLog)
                        .where(*success_conditions)
                    ).first()

                    error_conditions = filter_conditions + [
                        AccessLog.status_code >= 400,
                    ]
                    total_error_requests = session.exec(
                        select(func.count())
                        .select_from(AccessLog)
                        .where(*error_conditions)
                    ).first()

                    # Summary results are already computed individually above

                    # Get service type breakdown - simplified approach
                    service_breakdown = {}
                    # Get unique service types first
                    unique_services = session.exec(
                        select(AccessLog.service_type)
                        .distinct()
                        .where(*filter_conditions)
                    ).all()

                    # For each service type, get its statistics
                    for service in unique_services:
                        if service:  # Skip None values
                            # Build service-specific filter conditions
                            service_conditions = []
                            if start_dt:
                                service_conditions.append(
                                    AccessLog.timestamp >= start_dt
                                )
                            if end_dt:
                                service_conditions.append(AccessLog.timestamp <= end_dt)
                            if model:
                                service_conditions.append(AccessLog.model == model)
                            service_conditions.append(AccessLog.service_type == service)

                            service_count = session.exec(
                                select(func.count())
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_avg_duration = session.exec(
                                select(func.avg(AccessLog.duration_ms))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_total_cost = session.exec(
                                select(func.sum(AccessLog.cost_usd))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_total_tokens_input = session.exec(
                                select(func.sum(AccessLog.tokens_input))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_total_tokens_output = session.exec(
                                select(func.sum(AccessLog.tokens_output))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_cache_read_tokens = session.exec(
                                select(func.sum(AccessLog.cache_read_tokens))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_cache_write_tokens = session.exec(
                                select(func.sum(AccessLog.cache_write_tokens))
                                .select_from(AccessLog)
                                .where(*service_conditions)
                            ).first()

                            service_success_conditions = service_conditions + [
                                AccessLog.status_code >= 200,
                                AccessLog.status_code < 400,
                            ]
                            service_success_count = session.exec(
                                select(func.count())
                                .select_from(AccessLog)
                                .where(*service_success_conditions)
                            ).first()

                            service_error_conditions = service_conditions + [
                                AccessLog.status_code >= 400,
                            ]
                            service_error_count = session.exec(
                                select(func.count())
                                .select_from(AccessLog)
                                .where(*service_error_conditions)
                            ).first()

                            service_breakdown[service] = {
                                "request_count": service_count or 0,
                                "successful_requests": service_success_count or 0,
                                "error_requests": service_error_count or 0,
                                "success_rate": (service_success_count or 0)
                                / (service_count or 1)
                                * 100
                                if service_count
                                else 0,
                                "error_rate": (service_error_count or 0)
                                / (service_count or 1)
                                * 100
                                if service_count
                                else 0,
                                "avg_duration_ms": service_avg_duration or 0,
                                "total_cost_usd": service_total_cost or 0,
                                "total_tokens_input": service_total_tokens_input or 0,
                                "total_tokens_output": service_total_tokens_output or 0,
                                "total_cache_read_tokens": service_cache_read_tokens
                                or 0,
                                "total_cache_write_tokens": service_cache_write_tokens
                                or 0,
                                "total_tokens_all": (service_total_tokens_input or 0)
                                + (service_total_tokens_output or 0)
                                + (service_cache_read_tokens or 0)
                                + (service_cache_write_tokens or 0),
                            }

                    analytics = {
                        "summary": {
                            "total_requests": total_requests or 0,
                            "total_successful_requests": total_successful_requests or 0,
                            "total_error_requests": total_error_requests or 0,
                            "avg_duration_ms": avg_duration or 0,
                            "total_cost_usd": total_cost or 0,
                            "total_tokens_input": total_tokens_input or 0,
                            "total_tokens_output": total_tokens_output or 0,
                            "total_cache_read_tokens": total_cache_read_tokens or 0,
                            "total_cache_write_tokens": total_cache_write_tokens or 0,
                            "total_tokens_all": (total_tokens_input or 0)
                            + (total_tokens_output or 0)
                            + (total_cache_read_tokens or 0)
                            + (total_cache_write_tokens or 0),
                        },
                        "token_analytics": {
                            "input_tokens": total_tokens_input or 0,
                            "output_tokens": total_tokens_output or 0,
                            "cache_read_tokens": total_cache_read_tokens or 0,
                            "cache_write_tokens": total_cache_write_tokens or 0,
                            "total_tokens": (total_tokens_input or 0)
                            + (total_tokens_output or 0)
                            + (total_cache_read_tokens or 0)
                            + (total_cache_write_tokens or 0),
                        },
                        "request_analytics": {
                            "total_requests": total_requests or 0,
                            "successful_requests": total_successful_requests or 0,
                            "error_requests": total_error_requests or 0,
                            "success_rate": (total_successful_requests or 0)
                            / (total_requests or 1)
                            * 100
                            if total_requests
                            else 0,
                            "error_rate": (total_error_requests or 0)
                            / (total_requests or 1)
                            * 100
                            if total_requests
                            else 0,
                        },
                        "service_type_breakdown": service_breakdown,
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

    Returns a continuous stream of request events using event-driven SSE
    instead of polling. Events are emitted in real-time when requests
    start, complete, or error.
    """
    import asyncio
    import uuid
    from collections.abc import AsyncIterator

    async def event_stream() -> AsyncIterator[str]:
        """Generate Server-Sent Events for real-time metrics."""
        from ccproxy.observability.sse_events import get_sse_manager

        # Get SSE manager
        sse_manager = get_sse_manager()

        # Create unique connection ID
        connection_id = str(uuid.uuid4())

        try:
            # Use SSE manager for event-driven streaming
            async for event_data in sse_manager.add_connection(connection_id):
                yield event_data

        except asyncio.CancelledError:
            # Connection was cancelled, cleanup handled by SSE manager
            pass
        except Exception as e:
            # Send error event
            import json

            error_event = {
                "type": "error",
                "message": str(e),
                "timestamp": time.time(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

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
    service_type: str | None = Query(
        None,
        description="Filter by service type. Supports comma-separated values (e.g., 'proxy_service,sdk_service') and negation with ! prefix (e.g., '!access_log,!sdk_service')",
    ),
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

                    statement = select(AccessLog)

                    # Apply service type filtering with comma-separated values and negation
                    if service_type:
                        service_filters = [s.strip() for s in service_type.split(",")]
                        include_filters = [
                            f for f in service_filters if not f.startswith("!")
                        ]
                        exclude_filters = [
                            f[1:] for f in service_filters if f.startswith("!")
                        ]

                        if include_filters:
                            statement = statement.where(
                                AccessLog.service_type.in_(include_filters)
                            )
                        if exclude_filters:
                            statement = statement.where(
                                ~AccessLog.service_type.in_(exclude_filters)
                            )

                    statement = (
                        statement.order_by(order_clause).offset(offset).limit(limit)
                    )
                    results = session.exec(statement).all()

                    # Get total count with same filters
                    count_statement = select(func.count()).select_from(AccessLog)

                    # Apply same service type filtering to count
                    if service_type:
                        service_filters = [s.strip() for s in service_type.split(",")]
                        include_filters = [
                            f for f in service_filters if not f.startswith("!")
                        ]
                        exclude_filters = [
                            f[1:] for f in service_filters if f.startswith("!")
                        ]

                        if include_filters:
                            count_statement = count_statement.where(
                                AccessLog.service_type.in_(include_filters)
                            )
                        if exclude_filters:
                            count_statement = count_statement.where(
                                ~AccessLog.service_type.in_(exclude_filters)
                            )

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
                        "service_type": service_type,
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
