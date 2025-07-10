"""Metrics endpoints for Claude Code Proxy API Server."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from ccproxy.api.dependencies import get_metrics_service
from ccproxy.core.logging import get_logger
from ccproxy.metrics.models import AggregatedMetrics, MetricsSummary
from ccproxy.services.metrics_service import MetricsService


router = APIRouter()
logger = get_logger(__name__)


@router.get("/")
async def get_metrics_summary(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> MetricsSummary:
    """Get a summary of all metrics.

    Args:
        metrics_service: Injected metrics service dependency

    Returns:
        Summary of all collected metrics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug("Metrics summary request")
        summary = await metrics_service.get_summary()
        return summary

    except Exception as e:
        logger.error(f"Metrics summary request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/aggregated")
async def get_aggregated_metrics(
    time_range: Literal["1h", "24h", "7d", "30d"] = Query(
        "1h", description="Time range for aggregation"
    ),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> AggregatedMetrics:
    """Get aggregated metrics for a specific time range.

    Args:
        time_range: Time range for metric aggregation
        metrics_service: Injected metrics service dependency

    Returns:
        Aggregated metrics for the specified time range

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug(f"Aggregated metrics request: time_range={time_range}")
        aggregated = await metrics_service.get_aggregated(time_range)
        return aggregated

    except Exception as e:
        logger.error(f"Aggregated metrics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/requests")
async def get_request_metrics(
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, Any]:
    """Get request metrics with pagination.

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        metrics_service: Injected metrics service dependency

    Returns:
        List of request metrics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug(f"Request metrics request: limit={limit}, offset={offset}")
        metrics = await metrics_service.get_request_metrics(limit=limit, offset=offset)
        return metrics

    except Exception as e:
        logger.error(f"Request metrics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/costs")
async def get_cost_metrics(
    time_range: Literal["1h", "24h", "7d", "30d"] = Query(
        "24h", description="Time range for cost analysis"
    ),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, Any]:
    """Get cost metrics and analysis.

    Args:
        time_range: Time range for cost analysis
        metrics_service: Injected metrics service dependency

    Returns:
        Cost metrics and analysis

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug(f"Cost metrics request: time_range={time_range}")
        costs = await metrics_service.get_cost_metrics(time_range)
        return costs

    except Exception as e:
        logger.error(f"Cost metrics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/usage")
async def get_usage_metrics(
    time_range: Literal["1h", "24h", "7d", "30d"] = Query(
        "24h", description="Time range for usage analysis"
    ),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, Any]:
    """Get usage metrics and analysis.

    Args:
        time_range: Time range for usage analysis
        metrics_service: Injected metrics service dependency

    Returns:
        Usage metrics and analysis

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug(f"Usage metrics request: time_range={time_range}")
        usage = await metrics_service.get_usage_metrics(time_range)
        return usage

    except Exception as e:
        logger.error(f"Usage metrics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/errors")
async def get_error_metrics(
    limit: int = Query(100, description="Maximum number of records to return"),
    offset: int = Query(0, description="Number of records to skip"),
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, Any]:
    """Get error metrics with pagination.

    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        metrics_service: Injected metrics service dependency

    Returns:
        List of error metrics

    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        logger.debug(f"Error metrics request: limit={limit}, offset={offset}")
        errors = await metrics_service.get_error_metrics(limit=limit, offset=offset)
        return errors

    except Exception as e:
        logger.error(f"Error metrics request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/prometheus")
async def get_prometheus_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> str:
    """Get metrics in Prometheus format.

    Args:
        metrics_service: Injected metrics service dependency

    Returns:
        Metrics in Prometheus format

    Raises:
        HTTPException: If metrics export fails
    """
    try:
        logger.debug("Prometheus metrics request")
        prometheus_metrics = await metrics_service.export_prometheus()
        return prometheus_metrics

    except Exception as e:
        logger.error(f"Prometheus metrics export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/reset")
async def reset_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service),
) -> dict[str, Any]:
    """Reset all metrics data.

    Args:
        metrics_service: Injected metrics service dependency

    Returns:
        Reset confirmation

    Raises:
        HTTPException: If metrics reset fails
    """
    try:
        logger.info("Metrics reset request")
        await metrics_service.reset_metrics()
        return {"status": "success", "message": "All metrics have been reset"}

    except Exception as e:
        logger.error(f"Metrics reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
