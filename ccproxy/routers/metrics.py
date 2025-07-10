"""Metrics API endpoints with WebSocket support for live updates."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ccproxy.config import get_settings
from ccproxy.metrics.calculator import get_cost_calculator
from ccproxy.metrics.collector import get_metrics_collector
from ccproxy.metrics.sync_storage import get_sync_metrics_storage
from ccproxy.middleware.auth import get_auth_dependency
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)

router = APIRouter(tags=["metrics"])


class MetricsResponse(BaseModel):
    """Response model for current metrics."""

    timestamp: datetime = Field(description="Timestamp of metrics snapshot")
    active_requests: int = Field(description="Current active requests", ge=0)
    total_requests: int = Field(description="Total requests processed", ge=0)
    total_errors: int = Field(description="Total errors encountered", ge=0)
    avg_response_time: float = Field(
        description="Average response time in seconds", ge=0
    )
    total_cost: float = Field(description="Total cost in USD", ge=0)
    request_rates: dict[str, float] = Field(description="Request rates by API type")
    model_usage: dict[str, dict[str, Any]] = Field(description="Model usage statistics")


class HistoricalMetricsResponse(BaseModel):
    """Response model for historical metrics."""

    start_time: datetime = Field(description="Start time of the data range")
    end_time: datetime = Field(description="End time of the data range")
    total_records: int = Field(description="Total number of records", ge=0)
    metrics: list[dict[str, Any]] = Field(description="Historical metrics data")


class CostBreakdownResponse(BaseModel):
    """Response model for cost breakdown."""

    total_cost: float = Field(description="Total cost in USD", ge=0)
    by_model: dict[str, float] = Field(description="Cost breakdown by model")
    by_api_type: dict[str, float] = Field(description="Cost breakdown by API type")
    by_endpoint: dict[str, float] = Field(description="Cost breakdown by endpoint")
    period_start: datetime = Field(description="Start of the analysis period")
    period_end: datetime = Field(description="End of the analysis period")


class WebSocketConnection:
    """Manages WebSocket connections for live metrics updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self.update_interval = 5.0  # 5 seconds
        self._broadcast_task: asyncio.Task[None] | None = None

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

        # Start broadcasting if this is the first connection
        if len(self.active_connections) == 1:
            self._start_broadcasting()

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

        # Stop broadcasting if no connections remain
        if not self.active_connections:
            self._stop_broadcasting()

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    def _start_broadcasting(self) -> None:
        """Start the periodic broadcasting task."""
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def _stop_broadcasting(self) -> None:
        """Stop the periodic broadcasting task."""
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            self._broadcast_task = None

    async def _broadcast_loop(self) -> None:
        """Continuously broadcast metrics updates."""
        while self.active_connections:
            try:
                # Get current metrics
                metrics_data = await get_current_metrics_data()

                # Convert metrics to chart format
                chart_data = convert_metrics_to_chart_data(metrics_data)

                # Send both metrics and chart data
                message = json.dumps(
                    {
                        "type": "metrics_update",
                        "data": metrics_data,
                        "charts": chart_data,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

                await self.broadcast(message)
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(self.update_interval)


# Global WebSocket connection manager
connection_manager = WebSocketConnection()


# Global time series data store
_time_series_data: dict[str, list[tuple[datetime, float]]] = {
    "request_rate": [],
    "response_time": [],
    "error_rate": [],
    "active_requests": [],
}


def add_time_series_point(metric_name: str, value: float) -> None:
    """Add a time series data point."""
    current_time = datetime.utcnow()

    # Initialize if not exists
    if metric_name not in _time_series_data:
        _time_series_data[metric_name] = []

    # Add current point (avoid duplicate timestamps)
    data_points = _time_series_data[metric_name]
    if not data_points or (current_time - data_points[-1][0]).total_seconds() >= 10:
        data_points.append((current_time, value))
    else:
        # Update the most recent point if it's within 10 seconds
        data_points[-1] = (current_time, value)

    # Keep only last hour of data with cleanup
    cutoff_time = current_time - timedelta(hours=1)
    _time_series_data[metric_name] = [
        (timestamp, val) for timestamp, val in data_points if timestamp > cutoff_time
    ]


def get_time_series_data(
    metric_name: str, duration_minutes: int = 60
) -> list[tuple[datetime, float]]:
    """Get time series data for a metric."""
    if metric_name not in _time_series_data:
        return []

    current_time = datetime.utcnow()
    cutoff_time = current_time - timedelta(minutes=duration_minutes)

    return [
        (timestamp, value)
        for timestamp, value in _time_series_data[metric_name]
        if timestamp > cutoff_time
    ]


def convert_metrics_to_chart_data(metrics_data: dict[str, Any]) -> dict[str, Any]:
    """Convert metrics data to Chart.js format."""
    current_time = datetime.utcnow()

    # Add current metrics to time series
    request_rates = metrics_data.get("request_rates", {})
    total_rate = sum(request_rates.values())
    add_time_series_point("request_rate", total_rate)
    add_time_series_point(
        "response_time", metrics_data.get("avg_response_time", 0) * 1000
    )  # Convert to ms

    error_rate = (
        metrics_data.get("total_errors", 0)
        / max(metrics_data.get("total_requests", 1), 1)
    ) * 100
    add_time_series_point("error_rate", error_rate)
    add_time_series_point("active_requests", metrics_data.get("active_requests", 0))

    # Generate time labels for the last hour (every 5 minutes)
    time_labels = []
    time_points = []
    for i in range(12):  # 12 * 5 minutes = 60 minutes
        time_point = current_time - timedelta(
            minutes=(11 - i) * 5
        )  # From oldest to newest
        time_labels.append(time_point.strftime("%H:%M"))
        time_points.append(time_point)

    # Get historical data for charts (no mock data)
    request_rate_history = get_time_series_data("request_rate", 60) or []
    response_time_history = get_time_series_data("response_time", 60) or []
    error_rate_history = get_time_series_data("error_rate", 60) or []

    # Fill in data points for each time label with better interpolation
    def fill_time_series(
        history: list[tuple[datetime, float]], time_points: list[datetime]
    ) -> list[float | None]:
        data: list[float | None] = []

        # If no history, return null values for Chart.js to handle gracefully
        if not history:
            return [None] * len(time_points)

        for target_time in time_points:
            # Find the closest data point within a reasonable time window (10 minutes)
            closest_value = None
            min_diff = float("inf")

            for timestamp, value in history:
                diff = abs((timestamp - target_time).total_seconds())
                if diff < min_diff and diff < 600:  # Within 10 minutes
                    min_diff = diff
                    closest_value = value

            # If no close data point found, try to interpolate
            if closest_value is None:
                # Find the two closest points (before and after target_time)
                before_point = None
                after_point = None

                for timestamp, value in history:
                    if timestamp <= target_time:
                        if before_point is None or timestamp > before_point[0]:
                            before_point = (timestamp, value)
                    elif timestamp > target_time and (
                        after_point is None or timestamp < after_point[0]
                    ):
                        after_point = (timestamp, value)

                # Simple linear interpolation if we have both points
                if before_point and after_point:
                    time_diff = (after_point[0] - before_point[0]).total_seconds()
                    if time_diff > 0:
                        target_diff = (target_time - before_point[0]).total_seconds()
                        ratio = target_diff / time_diff
                        closest_value = (
                            before_point[1] + (after_point[1] - before_point[1]) * ratio
                        )
                elif before_point:
                    # Use the most recent previous value if within 20 minutes
                    if (target_time - before_point[0]).total_seconds() < 1200:
                        closest_value = before_point[1]

            data.append(closest_value)

        return data

    # Request rate chart data
    request_rate_data = {
        "labels": time_labels,
        "datasets": [
            {
                "label": "Total Requests/sec",
                "data": fill_time_series(request_rate_history, time_points),
                "borderColor": "rgb(54, 162, 235)",
                "backgroundColor": "rgba(54, 162, 235, 0.1)",
                "fill": True,
                "spanGaps": True,  # Handle null values gracefully
            }
        ],
    }

    # Add individual API types if they exist
    for i, (api_type, rate) in enumerate(request_rates.items()):
        if rate > 0:  # Only add if there's activity
            request_rate_data["datasets"].append(
                {
                    "label": api_type.title(),
                    "data": [rate]
                    * len(time_labels),  # Use current rate for simplicity
                    "borderColor": f"hsl({i * 60 + 120}, 70%, 50%)",
                    "backgroundColor": f"hsla({i * 60 + 120}, 70%, 50%, 0.1)",
                    "fill": False,
                    "spanGaps": True,
                }
            )

    # API distribution pie chart
    api_distribution_data = {
        "labels": list(request_rates.keys()) if request_rates else ["No Data"],
        "datasets": [
            {
                "data": list(request_rates.values()) if request_rates else [1],
                "backgroundColor": [
                    f"hsl({i * 60}, 70%, 50%)" for i in range(len(request_rates) or 1)
                ],
            }
        ],
    }

    # Response time chart
    response_time_data = {
        "labels": time_labels,
        "datasets": [
            {
                "label": "Avg Response Time (ms)",
                "data": fill_time_series(response_time_history, time_points),
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.1)",
                "fill": True,
                "spanGaps": True,
            },
        ],
    }

    # Error rate chart
    error_rate_data = {
        "labels": time_labels,
        "datasets": [
            {
                "label": "Error Rate %",
                "data": fill_time_series(error_rate_history, time_points),
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.1)",
                "fill": True,
                "spanGaps": True,
            }
        ],
    }

    # Cost breakdown chart
    model_usage = metrics_data.get("model_usage", {})
    cost_breakdown_data = {
        "labels": list(model_usage.keys()) if model_usage else ["No Data"],
        "datasets": [
            {
                "label": "Cost ($)",
                "data": [usage.get("cost", 0) for usage in model_usage.values()]
                if model_usage
                else [0],
                "backgroundColor": [
                    f"hsl({i * 45}, 70%, 50%)" for i in range(len(model_usage) or 1)
                ],
            }
        ],
    }

    # Token usage over time chart
    token_usage_data = {
        "labels": time_labels,
        "datasets": [
            {
                "label": "Input Tokens",
                "data": [metrics_data.get("tokenUsage", {}).get("inputTokens", 0)]
                * len(time_labels),
                "borderColor": "#007bff",
                "backgroundColor": "rgba(0, 123, 255, 0.1)",
                "fill": True,
                "spanGaps": True,
            },
            {
                "label": "Output Tokens",
                "data": [metrics_data.get("tokenUsage", {}).get("outputTokens", 0)]
                * len(time_labels),
                "borderColor": "#28a745",
                "backgroundColor": "rgba(40, 167, 69, 0.1)",
                "fill": True,
                "spanGaps": True,
            },
            {
                "label": "Cache Tokens",
                "data": [
                    (
                        metrics_data.get("tokenUsage", {}).get("cacheReadTokens", 0)
                        + metrics_data.get("tokenUsage", {}).get(
                            "cacheCreationTokens", 0
                        )
                    )
                ]
                * len(time_labels),
                "borderColor": "#ffc107",
                "backgroundColor": "rgba(255, 193, 7, 0.1)",
                "fill": True,
                "spanGaps": True,
            },
        ],
    }

    # Token usage by model chart
    token_by_model_data = {
        "labels": list(model_usage.keys()) if model_usage else ["No Data"],
        "datasets": [
            {
                "label": "Input Tokens",
                "data": [usage.get("input_tokens", 0) for usage in model_usage.values()]
                if model_usage
                else [0],
                "backgroundColor": "#007bff",
                "stack": "tokens",
            },
            {
                "label": "Output Tokens",
                "data": [
                    usage.get("output_tokens", 0) for usage in model_usage.values()
                ]
                if model_usage
                else [0],
                "backgroundColor": "#28a745",
                "stack": "tokens",
            },
        ],
    }

    # Cache token usage chart
    token_usage_stats = metrics_data.get("tokenUsage", {})
    cache_read = token_usage_stats.get("cacheReadTokens", 0)
    cache_creation = token_usage_stats.get("cacheCreationTokens", 0)
    regular_tokens = (
        token_usage_stats.get("totalTokens", 0) - cache_read - cache_creation
    )

    cache_token_data = {
        "labels": ["Cache Read", "Cache Creation", "No Cache"],
        "datasets": [
            {
                "data": [cache_read, cache_creation, max(0, regular_tokens)],
                "backgroundColor": ["#28a745", "#ffc107", "#6c757d"],
            }
        ],
    }

    return {
        "requestRateChart": request_rate_data,
        "responseTimeChart": response_time_data,
        "apiDistributionChart": api_distribution_data,
        "errorRateChart": error_rate_data,
        "costBreakdownChart": cost_breakdown_data,
        "tokenUsageChart": token_usage_data,
        "tokenByModelChart": token_by_model_data,
        "cacheTokenChart": cache_token_data,
    }


async def get_current_metrics_data() -> dict[str, Any]:
    """Get current metrics data for API responses and WebSocket broadcasts."""
    settings = get_settings()

    # Get the metrics collector to access Prometheus metrics
    metrics_collector = get_metrics_collector()

    # Get metrics storage if available
    storage = None
    if hasattr(settings, "metrics_enabled") and settings.metrics_enabled:
        try:
            storage = get_sync_metrics_storage(f"sqlite:///{settings.metrics_db_path}")
        except Exception as e:
            logger.warning(f"Failed to get metrics storage: {e}")

    # Get active requests from Prometheus metrics
    active_requests_count = 0
    try:
        # Access the Prometheus registry to get current gauge values
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            if metric.name == "ccproxy_active_requests":
                for sample in metric.samples:
                    # Sum all active requests across different API types
                    if sample.name == "ccproxy_active_requests":
                        active_requests_count += int(sample.value)
    except Exception as e:
        logger.debug(f"Failed to get active requests from Prometheus: {e}")

    # Default values when no data is available
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "active_requests": active_requests_count,
        "total_requests": 0,
        "total_errors": 0,
        "avg_response_time": 0,
        "total_cost": 0,
        "request_rates": {},
        "model_usage": {},
        "tokenUsage": {
            "totalTokens": 0,
            "inputTokens": 0,
            "outputTokens": 0,
            "cacheReadTokens": 0,
            "cacheCreationTokens": 0,
            "cacheHitRate": 0,
            "totalTokensChange": 0,
            "inputTokensChange": 0,
            "outputTokensChange": 0,
            "cacheHitRateChange": 0,
        },
    }

    if storage:
        try:
            # Get recent request logs (last hour)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            recent_logs = storage.get_request_logs(
                start_time=start_time, end_time=end_time, limit=1000
            )

            if recent_logs:
                # Calculate metrics
                total_requests = len(recent_logs)
                total_errors = sum(1 for log in recent_logs if log.status_code >= 400)
                avg_response_time = (
                    float(sum(log.duration_ms for log in recent_logs))
                    / total_requests
                    / 1000
                )
                total_cost = float(sum(log.cost_dollars for log in recent_logs))

                # Request rates by API type
                api_types: dict[str, int] = {}
                for log in recent_logs:
                    api_type = str(log.api_type)
                    api_types[api_type] = api_types.get(api_type, 0) + 1

                request_rates = {
                    api_type: count / 3600  # requests per second over last hour
                    for api_type, count in api_types.items()
                }

                # Model usage statistics
                model_usage = {}
                total_input_tokens = 0
                total_output_tokens = 0
                total_cache_read_tokens = 0
                total_cache_creation_tokens = 0

                for log in recent_logs:
                    if log.model:
                        if log.model not in model_usage:
                            model_usage[log.model] = {
                                "requests": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cache_read_tokens": 0,
                                "cache_creation_tokens": 0,
                                "cost": 0.0,
                            }
                        model_usage[log.model]["requests"] += 1
                        model_usage[log.model]["input_tokens"] += int(log.input_tokens)
                        model_usage[log.model]["output_tokens"] += int(
                            log.output_tokens
                        )
                        model_usage[log.model]["cache_read_tokens"] += int(
                            log.cache_read_input_tokens
                        )
                        model_usage[log.model]["cache_creation_tokens"] += int(
                            log.cache_creation_input_tokens
                        )
                        model_usage[log.model]["cost"] += float(log.cost_dollars)

                        # Accumulate totals
                        total_input_tokens += int(log.input_tokens)
                        total_output_tokens += int(log.output_tokens)
                        total_cache_read_tokens += int(log.cache_read_input_tokens)
                        total_cache_creation_tokens += int(
                            log.cache_creation_input_tokens
                        )

                # Calculate cache hit rate
                total_cache_tokens = (
                    total_cache_read_tokens + total_cache_creation_tokens
                )
                total_all_tokens = total_input_tokens + total_output_tokens
                cache_hit_rate = (
                    (total_cache_tokens / total_all_tokens * 100)
                    if total_all_tokens > 0
                    else 0
                )

                # Calculate change values by comparing with previous period
                # Get logs from previous hour for comparison
                prev_end_time = start_time
                prev_start_time = prev_end_time - timedelta(hours=1)

                prev_input_tokens = 0
                prev_output_tokens = 0
                prev_cache_tokens = 0

                try:
                    prev_logs = storage.get_request_logs(
                        start_time=prev_start_time, end_time=prev_end_time, limit=1000
                    )

                    if prev_logs:
                        for log in prev_logs:
                            if log.model:
                                prev_input_tokens += int(log.input_tokens)
                                prev_output_tokens += int(log.output_tokens)
                                prev_cache_tokens += int(
                                    log.cache_read_input_tokens
                                ) + int(log.cache_creation_input_tokens)

                    # Calculate changes
                    prev_total = prev_input_tokens + prev_output_tokens
                    prev_cache_rate = (
                        (prev_cache_tokens / prev_total * 100) if prev_total > 0 else 0
                    )

                    total_tokens_change = (
                        total_input_tokens + total_output_tokens
                    ) - prev_total
                    input_tokens_change = total_input_tokens - prev_input_tokens
                    output_tokens_change = total_output_tokens - prev_output_tokens
                    cache_hit_rate_change = cache_hit_rate - prev_cache_rate

                except Exception:
                    # If we can't get previous data, show 0 change
                    total_tokens_change = 0
                    input_tokens_change = 0
                    output_tokens_change = 0
                    cache_hit_rate_change = 0

                # Token usage statistics
                token_usage = {
                    "totalTokens": total_input_tokens + total_output_tokens,
                    "inputTokens": total_input_tokens,
                    "outputTokens": total_output_tokens,
                    "cacheReadTokens": total_cache_read_tokens,
                    "cacheCreationTokens": total_cache_creation_tokens,
                    "cacheHitRate": cache_hit_rate,
                    "totalTokensChange": total_tokens_change,
                    "inputTokensChange": input_tokens_change,
                    "outputTokensChange": output_tokens_change,
                    "cacheHitRateChange": cache_hit_rate_change,
                }

                metrics_data.update(
                    {
                        "active_requests": active_requests_count,  # Keep the real-time active requests
                        "total_requests": total_requests,
                        "total_errors": total_errors,
                        "avg_response_time": avg_response_time,
                        "total_cost": total_cost,
                        "request_rates": request_rates,
                        "model_usage": model_usage,
                        "tokenUsage": token_usage,
                    }
                )

        except Exception as e:
            logger.error(f"Error getting metrics data: {e}")

    return metrics_data


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(
    request: Request, auth_user: Any = Depends(get_auth_dependency)
) -> HTMLResponse:
    """Serve the metrics dashboard HTML page."""
    settings = get_settings()

    # Check if metrics are enabled
    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    # Serve the static dashboard HTML file
    from pathlib import Path

    dashboard_path = Path(__file__).parent.parent / "static" / "dashboard.html"

    if not dashboard_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard file not found")

    with open(dashboard_path) as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


@router.get("/dashboard_old", response_class=HTMLResponse)
async def get_dashboard_old(
    request: Request, auth_user: Any = Depends(get_auth_dependency)
) -> HTMLResponse:
    """Serve the old inline metrics dashboard HTML page."""
    settings = get_settings()

    # Check if metrics are enabled
    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    # Get WebSocket URL for live updates
    ws_url = str(request.url).replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.replace("/metrics/dashboard_old", "/metrics/ws")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Claude Proxy Metrics Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-title {{
                font-size: 14px;
                color: #666;
                margin-bottom: 10px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #333;
            }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }}
            .chart-container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                height: 500px;
                position: relative;
            }}
            .chart-container h3 {{
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 18px;
                color: #333;
            }}
            .chart-container canvas {{
                max-height: 420px;
            }}
            .status {{
                display: inline-block;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
            }}
            .status.connected {{
                background-color: #d4edda;
                color: #155724;
            }}
            .status.disconnected {{
                background-color: #f8d7da;
                color: #721c24;
            }}
            .last-updated {{
                color: #666;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Claude Proxy Metrics Dashboard</h1>
                <div>
                    <span class="status" id="connection-status">Connecting...</span>
                    <span class="last-updated" id="last-updated">Last updated: Never</span>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Active Requests</div>
                    <div class="metric-value" id="active-requests">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Requests (1h)</div>
                    <div class="metric-value" id="total-requests">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Errors (1h)</div>
                    <div class="metric-value" id="total-errors">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Avg Response Time</div>
                    <div class="metric-value" id="avg-response-time">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Cost (1h)</div>
                    <div class="metric-value" id="total-cost">-</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Request Rates by API Type</h3>
                    <canvas id="request-rates-chart"></canvas>
                </div>

                <div class="chart-container">
                    <h3>Model Usage</h3>
                    <canvas id="model-usage-chart"></canvas>
                </div>
            </div>
        </div>

        <script>
            // WebSocket connection
            const ws = new WebSocket('{ws_url}');
            const statusElement = document.getElementById('connection-status');
            const lastUpdatedElement = document.getElementById('last-updated');

            // Chart configurations
            const requestRatesChart = new Chart(document.getElementById('request-rates-chart'), {{
                type: 'doughnut',
                data: {{
                    labels: [],
                    datasets: [{{
                        data: [],
                        backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});

            const modelUsageChart = new Chart(document.getElementById('model-usage-chart'), {{
                type: 'bar',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Requests',
                        data: [],
                        backgroundColor: '#36A2EB'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});

            // WebSocket event handlers
            ws.onopen = function() {{
                statusElement.textContent = 'Connected';
                statusElement.className = 'status connected';
            }};

            ws.onclose = function() {{
                statusElement.textContent = 'Disconnected';
                statusElement.className = 'status disconnected';
            }};

            ws.onerror = function() {{
                statusElement.textContent = 'Error';
                statusElement.className = 'status disconnected';
            }};

            ws.onmessage = function(event) {{
                const message = JSON.parse(event.data);
                if (message.type === 'metrics_update') {{
                    updateMetrics(message.data);
                    lastUpdatedElement.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
                }}
            }};

            function updateMetrics(data) {{
                // Update metric cards
                document.getElementById('active-requests').textContent = data.active_requests;
                document.getElementById('total-requests').textContent = data.total_requests;
                document.getElementById('total-errors').textContent = data.total_errors;
                document.getElementById('avg-response-time').textContent = data.avg_response_time.toFixed(3) + 's';
                document.getElementById('total-cost').textContent = '$' + data.total_cost.toFixed(4);

                // Update request rates chart
                const rateLabels = Object.keys(data.request_rates);
                const rateData = Object.values(data.request_rates);
                requestRatesChart.data.labels = rateLabels;
                requestRatesChart.data.datasets[0].data = rateData;
                requestRatesChart.update();

                // Update model usage chart
                const modelLabels = Object.keys(data.model_usage);
                const modelData = modelLabels.map(model => data.model_usage[model].requests);

                // Handle empty model usage data
                if (modelLabels.length === 0) {{
                    modelUsageChart.data.labels = ["No Model Data"];
                    modelUsageChart.data.datasets[0].data = [0];
                }} else {{
                    modelUsageChart.data.labels = modelLabels;
                    modelUsageChart.data.datasets[0].data = modelData;
                }}
                modelUsageChart.update();
            }}

            // Cleanup on page unload
            window.addEventListener('beforeunload', function() {{
                ws.close();
            }});
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/api/current", response_model=MetricsResponse)
async def get_current_metrics(
    auth_user: Any = Depends(get_auth_dependency),
) -> MetricsResponse:
    """Get current metrics data as JSON."""
    settings = get_settings()

    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    metrics_data = await get_current_metrics_data()

    return MetricsResponse(
        timestamp=datetime.fromisoformat(metrics_data["timestamp"]),
        active_requests=metrics_data["active_requests"],
        total_requests=metrics_data["total_requests"],
        total_errors=metrics_data["total_errors"],
        avg_response_time=metrics_data["avg_response_time"],
        total_cost=metrics_data["total_cost"],
        request_rates=metrics_data["request_rates"],
        model_usage=metrics_data["model_usage"],
    )


@router.get("/api/history", response_model=HistoricalMetricsResponse)
async def get_historical_metrics(
    start_time: datetime | None = Query(None, description="Start time filter"),
    end_time: datetime | None = Query(None, description="End time filter"),
    endpoint: str | None = Query(None, description="Filter by endpoint"),
    api_type: str | None = Query(
        None, description="Filter by API type (anthropic, openai)"
    ),
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of records"),
    auth_user: Any = Depends(get_auth_dependency),
) -> HistoricalMetricsResponse:
    """Get historical metrics data with filtering options."""
    settings = get_settings()

    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    # Default to last 24 hours if no time range specified
    if not start_time:
        start_time = datetime.utcnow() - timedelta(days=1)
    if not end_time:
        end_time = datetime.utcnow()

    storage = get_sync_metrics_storage(f"sqlite:///{settings.metrics_db_path}")
    request_logs = storage.get_request_logs(
        start_time=start_time,
        end_time=end_time,
        endpoint=endpoint,
        api_type=api_type,
        limit=limit,
    )

    # Convert to JSON-serializable format
    metrics = []
    for log in request_logs:
        metrics.append(
            {
                "timestamp": log.timestamp.isoformat(),
                "method": log.method,
                "endpoint": log.endpoint,
                "api_type": log.api_type,
                "model": log.model,
                "status_code": log.status_code,
                "duration_ms": log.duration_ms,
                "request_size": log.request_size,
                "response_size": log.response_size,
                "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens,
                "cost_dollars": log.cost_dollars,
                "user_agent_category": log.user_agent_category,
                "error_type": log.error_type,
            }
        )

    return HistoricalMetricsResponse(
        start_time=start_time,
        end_time=end_time,
        total_records=len(metrics),
        metrics=metrics,
    )


@router.get("/api/costs", response_model=CostBreakdownResponse)
async def get_cost_breakdown(
    start_time: datetime | None = Query(None, description="Start time filter"),
    end_time: datetime | None = Query(None, description="End time filter"),
    auth_user: Any = Depends(get_auth_dependency),
) -> CostBreakdownResponse:
    """Get cost breakdown and analysis."""
    settings = get_settings()

    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    # Default to last 24 hours if no time range specified
    if not start_time:
        start_time = datetime.utcnow() - timedelta(days=1)
    if not end_time:
        end_time = datetime.utcnow()

    storage = get_sync_metrics_storage(f"sqlite:///{settings.metrics_db_path}")
    request_logs = storage.get_request_logs(
        start_time=start_time, end_time=end_time, limit=10000
    )

    # Calculate cost breakdowns
    total_cost = 0.0
    by_model: dict[str, float] = {}
    by_api_type: dict[str, float] = {}
    by_endpoint: dict[str, float] = {}

    for log in request_logs:
        cost = float(log.cost_dollars)
        total_cost += cost

        # By model
        if log.model:
            model = str(log.model)
            by_model[model] = by_model.get(model, 0.0) + cost

        # By API type
        api_type = str(log.api_type)
        by_api_type[api_type] = by_api_type.get(api_type, 0.0) + cost

        # By endpoint
        endpoint = str(log.endpoint)
        by_endpoint[endpoint] = by_endpoint.get(endpoint, 0.0) + cost

    return CostBreakdownResponse(
        total_cost=total_cost,
        by_model=by_model,
        by_api_type=by_api_type,
        by_endpoint=by_endpoint,
        period_start=start_time,
        period_end=end_time,
    )


@router.get("/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics(
    auth_user: Any = Depends(get_auth_dependency),
) -> PlainTextResponse:
    """Export metrics in Prometheus format."""
    settings = get_settings()

    if not getattr(settings, "metrics_enabled", False):
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    # Generate Prometheus metrics
    metrics_data = generate_latest()

    return PlainTextResponse(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time metrics updates."""
    settings = get_settings()

    if not getattr(settings, "metrics_enabled", False):
        await websocket.close(code=1008, reason="Metrics not enabled")
        return

    await connection_manager.connect(websocket)

    try:
        while True:
            # Keep the connection alive by waiting for messages
            # In a real implementation, you might want to handle client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)
