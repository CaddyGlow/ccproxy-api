"""
Stats collector and printer for periodic metrics summary.

This module provides functionality to collect and print periodic statistics
from the observability system, including Prometheus metrics and DuckDB storage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from ccproxy.config.observability import ObservabilitySettings


logger = structlog.get_logger(__name__)


@dataclass
class StatsSnapshot:
    """Snapshot of current statistics."""

    timestamp: datetime
    requests_total: int
    requests_last_minute: int
    avg_response_time_ms: float
    avg_response_time_last_minute_ms: float
    tokens_input_total: int
    tokens_output_total: int
    tokens_input_last_minute: int
    tokens_output_last_minute: int
    cost_total_usd: float
    cost_last_minute_usd: float
    errors_total: int
    errors_last_minute: int
    active_requests: int
    top_model: str
    top_model_percentage: float


class StatsCollector:
    """
    Collects and formats metrics statistics for periodic printing.

    Integrates with both Prometheus metrics and DuckDB storage to provide
    comprehensive statistics about the API performance.
    """

    def __init__(
        self,
        settings: ObservabilitySettings,
        metrics_instance: Any | None = None,
        storage_instance: Any | None = None,
    ):
        """
        Initialize stats collector.

        Args:
            settings: Observability configuration settings
            metrics_instance: Prometheus metrics instance
            storage_instance: DuckDB storage instance
        """
        self.settings = settings
        self._metrics_instance = metrics_instance
        self._storage_instance = storage_instance
        self._last_snapshot: StatsSnapshot | None = None
        self._last_collection_time = time.time()

    async def collect_stats(self) -> StatsSnapshot:
        """
        Collect current statistics from all available sources.

        Returns:
            StatsSnapshot with current metrics
        """
        current_time = time.time()
        timestamp = datetime.now()

        # Initialize default values
        stats_data: dict[str, Any] = {
            "timestamp": timestamp,
            "requests_total": 0,
            "requests_last_minute": 0,
            "avg_response_time_ms": 0.0,
            "avg_response_time_last_minute_ms": 0.0,
            "tokens_input_total": 0,
            "tokens_output_total": 0,
            "tokens_input_last_minute": 0,
            "tokens_output_last_minute": 0,
            "cost_total_usd": 0.0,
            "cost_last_minute_usd": 0.0,
            "errors_total": 0,
            "errors_last_minute": 0,
            "active_requests": 0,
            "top_model": "unknown",
            "top_model_percentage": 0.0,
        }

        # Collect from Prometheus metrics if available
        if self._metrics_instance and self._metrics_instance.is_enabled():
            try:
                await self._collect_from_prometheus(stats_data)
            except Exception as e:
                logger.warning(
                    "Failed to collect from Prometheus metrics", error=str(e)
                )

        # Collect from DuckDB storage if available
        if self._storage_instance and self._storage_instance.is_enabled():
            try:
                await self._collect_from_duckdb(stats_data, current_time)
            except Exception as e:
                logger.warning("Failed to collect from DuckDB storage", error=str(e))

        snapshot = StatsSnapshot(
            timestamp=stats_data["timestamp"],
            requests_total=int(stats_data["requests_total"]),
            requests_last_minute=int(stats_data["requests_last_minute"]),
            avg_response_time_ms=float(stats_data["avg_response_time_ms"]),
            avg_response_time_last_minute_ms=float(
                stats_data["avg_response_time_last_minute_ms"]
            ),
            tokens_input_total=int(stats_data["tokens_input_total"]),
            tokens_output_total=int(stats_data["tokens_output_total"]),
            tokens_input_last_minute=int(stats_data["tokens_input_last_minute"]),
            tokens_output_last_minute=int(stats_data["tokens_output_last_minute"]),
            cost_total_usd=float(stats_data["cost_total_usd"]),
            cost_last_minute_usd=float(stats_data["cost_last_minute_usd"]),
            errors_total=int(stats_data["errors_total"]),
            errors_last_minute=int(stats_data["errors_last_minute"]),
            active_requests=int(stats_data["active_requests"]),
            top_model=str(stats_data["top_model"]),
            top_model_percentage=float(stats_data["top_model_percentage"]),
        )
        self._last_snapshot = snapshot
        self._last_collection_time = current_time

        return snapshot

    async def _collect_from_prometheus(self, stats_data: dict[str, Any]) -> None:
        """Collect statistics from Prometheus metrics."""
        if not self._metrics_instance:
            return

        try:
            logger.debug(
                "prometheus_collection_starting",
                metrics_available=bool(self._metrics_instance),
            )

            # Get active requests from gauge
            if hasattr(self._metrics_instance, "active_requests"):
                active_value = self._metrics_instance.active_requests._value._value
                stats_data["active_requests"] = int(active_value)
                logger.debug(
                    "prometheus_active_requests_collected", active_requests=active_value
                )

            # Get request counts from counter
            if hasattr(self._metrics_instance, "request_counter"):
                request_counter = self._metrics_instance.request_counter
                # Sum all request counts across all labels
                total_requests = 0
                for metric in request_counter.collect():
                    for sample in metric.samples:
                        if sample.name.endswith("_total"):
                            total_requests += sample.value
                stats_data["requests_total"] = int(total_requests)

                # Calculate last minute requests (difference from last snapshot)
                if self._last_snapshot:
                    last_minute_requests = (
                        total_requests - self._last_snapshot.requests_total
                    )
                    stats_data["requests_last_minute"] = max(
                        0, int(last_minute_requests)
                    )
                else:
                    stats_data["requests_last_minute"] = int(total_requests)

                logger.debug(
                    "prometheus_requests_collected",
                    total_requests=total_requests,
                    requests_last_minute=stats_data["requests_last_minute"],
                )

            # Get response times from histogram
            if hasattr(self._metrics_instance, "response_time"):
                response_time = self._metrics_instance.response_time
                # Get total count and sum for average calculation
                total_count = 0
                total_sum = 0
                for metric in response_time.collect():
                    for sample in metric.samples:
                        if sample.name.endswith("_count"):
                            total_count += sample.value
                        elif sample.name.endswith("_sum"):
                            total_sum += sample.value

                if total_count > 0:
                    avg_response_time_seconds = total_sum / total_count
                    stats_data["avg_response_time_ms"] = (
                        avg_response_time_seconds * 1000
                    )

                    # Calculate last minute average response time
                    if self._last_snapshot and self._last_snapshot.requests_total > 0:
                        last_minute_count = (
                            total_count - self._last_snapshot.requests_total
                        )
                        if last_minute_count > 0:
                            # Calculate the sum for just the last minute
                            last_minute_sum = total_sum - (
                                self._last_snapshot.requests_total
                                * self._last_snapshot.avg_response_time_ms
                                / 1000
                            )
                            last_minute_avg = (
                                last_minute_sum / last_minute_count
                            ) * 1000
                            stats_data["avg_response_time_last_minute_ms"] = float(
                                last_minute_avg
                            )
                        else:
                            stats_data["avg_response_time_last_minute_ms"] = 0.0
                    else:
                        stats_data["avg_response_time_last_minute_ms"] = stats_data[
                            "avg_response_time_ms"
                        ]

            # Get token counts from counter
            if hasattr(self._metrics_instance, "token_counter"):
                token_counter = self._metrics_instance.token_counter
                tokens_input = 0
                tokens_output = 0
                for metric in token_counter.collect():
                    for sample in metric.samples:
                        if sample.name.endswith("_total"):
                            token_type = sample.labels.get("type", "")
                            if token_type == "input":
                                tokens_input += sample.value
                            elif token_type == "output":
                                tokens_output += sample.value

                stats_data["tokens_input_total"] = int(tokens_input)
                stats_data["tokens_output_total"] = int(tokens_output)

                # Calculate last minute tokens
                if self._last_snapshot:
                    last_minute_input = (
                        tokens_input - self._last_snapshot.tokens_input_total
                    )
                    last_minute_output = (
                        tokens_output - self._last_snapshot.tokens_output_total
                    )
                    stats_data["tokens_input_last_minute"] = max(
                        0, int(last_minute_input)
                    )
                    stats_data["tokens_output_last_minute"] = max(
                        0, int(last_minute_output)
                    )
                else:
                    stats_data["tokens_input_last_minute"] = int(tokens_input)
                    stats_data["tokens_output_last_minute"] = int(tokens_output)

            # Get cost from counter
            if hasattr(self._metrics_instance, "cost_counter"):
                cost_counter = self._metrics_instance.cost_counter
                total_cost = 0
                for metric in cost_counter.collect():
                    for sample in metric.samples:
                        if sample.name.endswith("_total"):
                            total_cost += sample.value
                stats_data["cost_total_usd"] = float(total_cost)

                # Calculate last minute cost
                if self._last_snapshot:
                    last_minute_cost = total_cost - self._last_snapshot.cost_total_usd
                    stats_data["cost_last_minute_usd"] = max(
                        0.0, float(last_minute_cost)
                    )
                else:
                    stats_data["cost_last_minute_usd"] = float(total_cost)

            # Get error counts from counter
            if hasattr(self._metrics_instance, "error_counter"):
                error_counter = self._metrics_instance.error_counter
                total_errors = 0
                for metric in error_counter.collect():
                    for sample in metric.samples:
                        if sample.name.endswith("_total"):
                            total_errors += sample.value
                stats_data["errors_total"] = int(total_errors)

                # Calculate last minute errors
                if self._last_snapshot:
                    last_minute_errors = total_errors - self._last_snapshot.errors_total
                    stats_data["errors_last_minute"] = max(0, int(last_minute_errors))
                else:
                    stats_data["errors_last_minute"] = int(total_errors)

            logger.debug(
                "prometheus_stats_collected",
                requests_total=stats_data["requests_total"],
                requests_last_minute=stats_data["requests_last_minute"],
                avg_response_time_ms=stats_data["avg_response_time_ms"],
                tokens_input_total=stats_data["tokens_input_total"],
                tokens_output_total=stats_data["tokens_output_total"],
                cost_total_usd=stats_data["cost_total_usd"],
                errors_total=stats_data["errors_total"],
                active_requests=stats_data["active_requests"],
            )

        except Exception as e:
            logger.debug("Failed to get metrics from Prometheus", error=str(e))

    async def _collect_from_duckdb(
        self, stats_data: dict[str, Any], current_time: float
    ) -> None:
        """Collect statistics from DuckDB storage."""
        if not self._storage_instance:
            return

        try:
            # Get overall analytics
            overall_analytics = await self._storage_instance.get_analytics()
            if overall_analytics and "summary" in overall_analytics:
                summary = overall_analytics["summary"]
                stats_data["requests_total"] = summary.get("total_requests", 0)
                stats_data["avg_response_time_ms"] = summary.get("avg_duration_ms", 0.0)
                stats_data["tokens_input_total"] = summary.get("total_tokens_input", 0)
                stats_data["tokens_output_total"] = summary.get(
                    "total_tokens_output", 0
                )
                stats_data["cost_total_usd"] = summary.get("total_cost_usd", 0.0)

            # Get last minute analytics
            one_minute_ago = current_time - 60
            last_minute_analytics = await self._storage_instance.get_analytics(
                start_time=one_minute_ago,
                end_time=current_time,
            )

            if last_minute_analytics and "summary" in last_minute_analytics:
                last_minute_summary = last_minute_analytics["summary"]
                stats_data["requests_last_minute"] = last_minute_summary.get(
                    "total_requests", 0
                )
                stats_data["avg_response_time_last_minute_ms"] = (
                    last_minute_summary.get("avg_duration_ms", 0.0)
                )
                stats_data["tokens_input_last_minute"] = last_minute_summary.get(
                    "total_tokens_input", 0
                )
                stats_data["tokens_output_last_minute"] = last_minute_summary.get(
                    "total_tokens_output", 0
                )
                stats_data["cost_last_minute_usd"] = last_minute_summary.get(
                    "total_cost_usd", 0.0
                )

            # Get top model from last minute data
            await self._get_top_model(stats_data, one_minute_ago, current_time)

        except Exception as e:
            logger.debug("Failed to collect from DuckDB", error=str(e))

    async def _get_top_model(
        self, stats_data: dict[str, Any], start_time: float, end_time: float
    ) -> None:
        """Get the most used model in the time period."""
        if not self._storage_instance:
            return

        try:
            # Query for model usage
            sql = """
                SELECT model, COUNT(*) as request_count
                FROM access_logs
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY model
                ORDER BY request_count DESC
                LIMIT 1
            """

            start_dt = datetime.fromtimestamp(start_time)
            end_dt = datetime.fromtimestamp(end_time)

            results = await self._storage_instance.query(
                sql, [start_dt, end_dt], limit=1
            )

            if results:
                top_model_data = results[0]
                stats_data["top_model"] = top_model_data.get("model", "unknown")
                request_count = top_model_data.get("request_count", 0)

                if stats_data["requests_last_minute"] > 0:
                    stats_data["top_model_percentage"] = (
                        request_count / stats_data["requests_last_minute"]
                    ) * 100
                else:
                    stats_data["top_model_percentage"] = 0.0

        except Exception as e:
            logger.debug("Failed to get top model", error=str(e))

    def _has_meaningful_activity(self, snapshot: StatsSnapshot) -> bool:
        """
        Check if there is meaningful activity to report.

        Args:
            snapshot: Stats snapshot to check

        Returns:
            True if there is meaningful activity, False otherwise
        """
        # Show stats if there are requests in the last minute
        if snapshot.requests_last_minute > 0:
            return True

        # Show stats if there are currently active requests
        if snapshot.active_requests > 0:
            return True

        # Show stats if there are any errors in the last minute
        if snapshot.errors_last_minute > 0:
            return True

        # Show stats if there are any total requests (for the first time)
        return snapshot.requests_total > 0 and self._last_snapshot is None

    def format_stats(self, snapshot: StatsSnapshot) -> str:
        """
        Format stats snapshot for display.

        Args:
            snapshot: Stats snapshot to format

        Returns:
            Formatted stats string
        """
        format_type = self.settings.stats_printing_format

        if format_type == "json":
            return self._format_json(snapshot)
        elif format_type == "rich":
            return self._format_rich(snapshot)
        elif format_type == "log":
            return self._format_log(snapshot)
        else:  # console (default)
            return self._format_console(snapshot)

    def _format_console(self, snapshot: StatsSnapshot) -> str:
        """Format stats for console output."""
        timestamp_str = snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Format response times
        avg_response_str = f"{snapshot.avg_response_time_ms:.1f}ms"
        avg_response_last_min_str = f"{snapshot.avg_response_time_last_minute_ms:.1f}ms"

        # Format costs
        cost_total_str = f"${snapshot.cost_total_usd:.4f}"
        cost_last_min_str = f"${snapshot.cost_last_minute_usd:.4f}"

        # Format top model percentage
        top_model_str = f"{snapshot.top_model} ({snapshot.top_model_percentage:.1f}%)"

        return f"""[{timestamp_str}] METRICS SUMMARY
├─ Requests: {snapshot.requests_last_minute} (last min) / {snapshot.requests_total} (total)
├─ Avg Response: {avg_response_last_min_str} (last min) / {avg_response_str} (overall)
├─ Tokens: {snapshot.tokens_input_last_minute:,} in / {snapshot.tokens_output_last_minute:,} out (last min)
├─ Cost: {cost_last_min_str} (last min) / {cost_total_str} (total)
├─ Errors: {snapshot.errors_last_minute} (last min) / {snapshot.errors_total} (total)
├─ Active: {snapshot.active_requests} requests
└─ Top Model: {top_model_str}"""

    def _format_json(self, snapshot: StatsSnapshot) -> str:
        """Format stats for JSON output."""
        data = {
            "timestamp": snapshot.timestamp.isoformat(),
            "requests": {
                "last_minute": snapshot.requests_last_minute,
                "total": snapshot.requests_total,
            },
            "response_time_ms": {
                "last_minute": snapshot.avg_response_time_last_minute_ms,
                "overall": snapshot.avg_response_time_ms,
            },
            "tokens": {
                "input_last_minute": snapshot.tokens_input_last_minute,
                "output_last_minute": snapshot.tokens_output_last_minute,
                "input_total": snapshot.tokens_input_total,
                "output_total": snapshot.tokens_output_total,
            },
            "cost_usd": {
                "last_minute": snapshot.cost_last_minute_usd,
                "total": snapshot.cost_total_usd,
            },
            "errors": {
                "last_minute": snapshot.errors_last_minute,
                "total": snapshot.errors_total,
            },
            "active_requests": snapshot.active_requests,
            "top_model": {
                "name": snapshot.top_model,
                "percentage": snapshot.top_model_percentage,
            },
        }
        return json.dumps(data, indent=2)

    def _format_rich(self, snapshot: StatsSnapshot) -> str:
        """Format stats for rich console output with colors and styling."""
        try:
            # Try to import rich for enhanced formatting
            from io import StringIO

            from rich import box
            from rich.console import Console
            from rich.table import Table

            output_buffer = StringIO()
            console = Console(file=output_buffer, width=80, force_terminal=True)
            timestamp_str = snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Create main stats table
            table = Table(title=f"METRICS SUMMARY - {timestamp_str}", box=box.ROUNDED)
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Last Minute", style="yellow", justify="right")
            table.add_column("Total", style="green", justify="right")

            # Add rows with formatted data
            table.add_row(
                "Requests",
                f"{snapshot.requests_last_minute:,}",
                f"{snapshot.requests_total:,}",
            )

            table.add_row(
                "Avg Response",
                f"{snapshot.avg_response_time_last_minute_ms:.1f}ms",
                f"{snapshot.avg_response_time_ms:.1f}ms",
            )

            table.add_row(
                "Tokens In",
                f"{snapshot.tokens_input_last_minute:,}",
                f"{snapshot.tokens_input_total:,}",
            )

            table.add_row(
                "Tokens Out",
                f"{snapshot.tokens_output_last_minute:,}",
                f"{snapshot.tokens_output_total:,}",
            )

            table.add_row(
                "Cost",
                f"${snapshot.cost_last_minute_usd:.4f}",
                f"${snapshot.cost_total_usd:.4f}",
            )

            table.add_row(
                "Errors",
                f"{snapshot.errors_last_minute}",
                f"{snapshot.errors_total}",
            )

            # Add single-column rows
            table.add_row("", "", "")  # Separator
            table.add_row("Active Requests", f"{snapshot.active_requests}", "")

            table.add_row(
                "Top Model",
                f"{snapshot.top_model}",
                f"({snapshot.top_model_percentage:.1f}%)",
            )

            console.print(table)
            output = output_buffer.getvalue()
            output_buffer.close()

            return output.strip()

        except ImportError:
            # Fallback to console format if rich is not available
            logger.warning("Rich not available, falling back to console format")
            return self._format_console(snapshot)
        except Exception as e:
            logger.warning(
                f"Rich formatting failed: {e}, falling back to console format"
            )
            return self._format_console(snapshot)

    def _format_log(self, snapshot: StatsSnapshot) -> str:
        """Format stats for structured logging output."""
        timestamp_str = snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Create a structured log entry
        log_data = {
            "timestamp": timestamp_str,
            "event": "metrics_summary",
            "requests": {
                "last_minute": snapshot.requests_last_minute,
                "total": snapshot.requests_total,
            },
            "response_time_ms": {
                "last_minute_avg": snapshot.avg_response_time_last_minute_ms,
                "overall_avg": snapshot.avg_response_time_ms,
            },
            "tokens": {
                "input_last_minute": snapshot.tokens_input_last_minute,
                "output_last_minute": snapshot.tokens_output_last_minute,
                "input_total": snapshot.tokens_input_total,
                "output_total": snapshot.tokens_output_total,
            },
            "cost_usd": {
                "last_minute": snapshot.cost_last_minute_usd,
                "total": snapshot.cost_total_usd,
            },
            "errors": {
                "last_minute": snapshot.errors_last_minute,
                "total": snapshot.errors_total,
            },
            "active_requests": snapshot.active_requests,
            "top_model": {
                "name": snapshot.top_model,
                "percentage": snapshot.top_model_percentage,
            },
        }

        # Format as a log line with key=value pairs
        log_parts = [f"[{timestamp_str}]", "event=metrics_summary"]

        log_parts.extend(
            [
                f"requests_last_min={snapshot.requests_last_minute}",
                f"requests_total={snapshot.requests_total}",
                f"avg_response_ms={snapshot.avg_response_time_ms:.1f}",
                f"avg_response_last_min_ms={snapshot.avg_response_time_last_minute_ms:.1f}",
                f"tokens_in_last_min={snapshot.tokens_input_last_minute}",
                f"tokens_out_last_min={snapshot.tokens_output_last_minute}",
                f"tokens_in_total={snapshot.tokens_input_total}",
                f"tokens_out_total={snapshot.tokens_output_total}",
                f"cost_last_min_usd={snapshot.cost_last_minute_usd:.4f}",
                f"cost_total_usd={snapshot.cost_total_usd:.4f}",
                f"errors_last_min={snapshot.errors_last_minute}",
                f"errors_total={snapshot.errors_total}",
                f"active_requests={snapshot.active_requests}",
                f"top_model={snapshot.top_model}",
                f"top_model_pct={snapshot.top_model_percentage:.1f}",
            ]
        )

        return " ".join(log_parts)

    async def print_stats(self) -> None:
        """Collect and print current statistics."""
        try:
            snapshot = await self.collect_stats()

            # Only print stats if there is meaningful activity
            if self._has_meaningful_activity(snapshot):
                formatted_stats = self.format_stats(snapshot)

                # Print to stdout for console visibility
                print(formatted_stats)

                # Also log for structured logging
                logger.info(
                    "stats_printed",
                    requests_last_minute=snapshot.requests_last_minute,
                    requests_total=snapshot.requests_total,
                    avg_response_time_ms=snapshot.avg_response_time_ms,
                    cost_total_usd=snapshot.cost_total_usd,
                    active_requests=snapshot.active_requests,
                    top_model=snapshot.top_model,
                )
            else:
                logger.debug(
                    "stats_skipped_no_activity",
                    requests_last_minute=snapshot.requests_last_minute,
                    requests_total=snapshot.requests_total,
                    active_requests=snapshot.active_requests,
                )

        except Exception as e:
            logger.error("Failed to print stats", error=str(e), exc_info=True)


# Global stats collector instance
_global_stats_collector: StatsCollector | None = None


def get_stats_collector(
    settings: ObservabilitySettings | None = None,
    metrics_instance: Any | None = None,
    storage_instance: Any | None = None,
) -> StatsCollector:
    """
    Get or create global stats collector instance.

    Args:
        settings: Observability settings
        metrics_instance: Metrics instance for dependency injection
        storage_instance: Storage instance for dependency injection

    Returns:
        StatsCollector instance
    """
    global _global_stats_collector

    if _global_stats_collector is None:
        if settings is None:
            from ccproxy.config.settings import get_settings

            settings = get_settings().observability

        if metrics_instance is None:
            try:
                from .metrics import get_metrics

                metrics_instance = get_metrics()
            except Exception as e:
                logger.warning("Failed to get metrics instance", error=str(e))

        if storage_instance is None:
            try:
                from .storage.duckdb_simple import SimpleDuckDBStorage

                storage_instance = SimpleDuckDBStorage(settings.duckdb_path)
                # Note: Storage needs to be initialized before use
            except Exception as e:
                logger.warning("Failed to get storage instance", error=str(e))

        _global_stats_collector = StatsCollector(
            settings=settings,
            metrics_instance=metrics_instance,
            storage_instance=storage_instance,
        )

    return _global_stats_collector


def reset_stats_collector() -> None:
    """Reset global stats collector instance (mainly for testing)."""
    global _global_stats_collector
    _global_stats_collector = None
