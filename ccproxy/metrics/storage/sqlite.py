"""
SQLite storage implementation for metrics.

This module provides a SQLite-based storage backend for metrics,
suitable for production deployments with persistent storage needs.
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import aiosqlite

from ..models import (
    CostMetric,
    ErrorMetric,
    LatencyMetric,
    MetricRecord,
    MetricsSummary,
    MetricType,
    RequestMetric,
    ResponseMetric,
    UsageMetric,
)
from .base import (
    MetricsStorage,
    StorageConnectionError,
    StorageError,
    StorageInitializationError,
    StorageOperationError,
)


logger = logging.getLogger(__name__)


class SQLiteMetricsStorage(MetricsStorage):
    """
    SQLite storage implementation for metrics.

    This storage backend uses SQLite for persistent storage with
    good performance and ACID properties.
    """

    SCHEMA_VERSION = "1.0.0"

    # SQL schema
    CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS schema_info (
        version TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS metrics (
        id TEXT PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        metric_type TEXT NOT NULL,
        request_id TEXT,
        user_id TEXT,
        session_id TEXT,
        metadata TEXT,  -- JSON

        -- Request fields
        method TEXT,
        path TEXT,
        endpoint TEXT,
        api_version TEXT,
        client_ip TEXT,
        user_agent TEXT,
        content_length INTEGER,
        content_type TEXT,
        model TEXT,
        provider TEXT,
        max_tokens INTEGER,
        temperature REAL,
        streaming BOOLEAN,

        -- Response fields
        status_code INTEGER,
        response_time_ms REAL,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cache_read_tokens INTEGER,
        cache_write_tokens INTEGER,
        first_token_time_ms REAL,
        stream_completion_time_ms REAL,
        completion_reason TEXT,
        safety_filtered BOOLEAN,

        -- Error fields
        error_type TEXT,
        error_code TEXT,
        error_message TEXT,
        stack_trace TEXT,
        retry_count INTEGER,
        recoverable BOOLEAN,

        -- Cost fields
        input_cost REAL,
        output_cost REAL,
        cache_read_cost REAL,
        cache_write_cost REAL,
        total_cost REAL,
        pricing_tier TEXT,
        currency TEXT,

        -- Latency fields
        request_processing_ms REAL,
        claude_api_call_ms REAL,
        response_processing_ms REAL,
        total_latency_ms REAL,
        queue_time_ms REAL,
        wait_time_ms REAL,
        first_token_latency_ms REAL,
        token_generation_rate REAL,

        -- Usage fields
        request_count INTEGER,
        token_count INTEGER,
        window_start TIMESTAMP,
        window_end TIMESTAMP,
        window_duration_seconds REAL,
        aggregation_level TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
    CREATE INDEX IF NOT EXISTS idx_metrics_user_id ON metrics(user_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics(session_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_request_id ON metrics(request_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics(model);
    CREATE INDEX IF NOT EXISTS idx_metrics_provider ON metrics(provider);
    CREATE INDEX IF NOT EXISTS idx_metrics_composite ON metrics(timestamp, metric_type, user_id);
    """

    def __init__(
        self,
        database_path: str = "metrics.db",
        connection_timeout: float = 30.0,
        enable_wal: bool = True,
        pragmas: dict[str, Any] | None = None,
    ):
        """
        Initialize the SQLite storage.

        Args:
            database_path: Path to the SQLite database file
            connection_timeout: Connection timeout in seconds
            enable_wal: Whether to enable WAL mode for better concurrency
            pragmas: Additional SQLite PRAGMA settings
        """
        self.database_path = Path(database_path)
        self.connection_timeout = connection_timeout
        self.enable_wal = enable_wal
        self.pragmas = pragmas or {}

        # Connection management
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

        # Performance tracking
        self._total_operations = 0
        self._failed_operations = 0

    async def initialize(self) -> None:
        """Initialize the SQLite storage."""
        try:
            # Create database directory if it doesn't exist
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Open connection
            self._connection = await aiosqlite.connect(
                str(self.database_path), timeout=self.connection_timeout
            )

            # Set pragmas
            if self.enable_wal:
                await self._connection.execute("PRAGMA journal_mode=WAL")

            # Apply custom pragmas
            for pragma, value in self.pragmas.items():
                await self._connection.execute(f"PRAGMA {pragma}={value}")

            # Create tables
            await self._connection.executescript(self.CREATE_TABLES_SQL)

            # Initialize schema version
            await self._ensure_schema_version()

            # Commit changes
            await self._connection.commit()

            logger.info(f"Initialized SQLite metrics storage at {self.database_path}")

        except Exception as e:
            error_msg = f"Failed to initialize SQLite storage: {e}"
            logger.error(error_msg)
            raise StorageInitializationError(error_msg) from e

    async def close(self) -> None:
        """Close the SQLite storage."""
        if self._connection:
            await self._connection.close()
            self._connection = None

        logger.info("Closed SQLite metrics storage")

    def _ensure_connection(self) -> None:
        """Ensure connection is available."""
        if self._connection is None:
            raise StorageConnectionError(
                "Database connection not initialized. Call initialize() first."
            )

    async def store_metric(self, metric: MetricRecord) -> bool:
        """Store a single metric record."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            async with self._lock:
                await self._insert_metric(metric)
                await self._connection.commit()
                self._total_operations += 1
            return True

        except Exception as e:
            self._failed_operations += 1
            logger.error(f"Failed to store metric: {e}")
            return False

    async def store_metrics(self, metrics: list[MetricRecord]) -> int:
        """Store multiple metric records."""
        if not metrics:
            return 0

        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        stored_count = 0

        try:
            async with self._lock:
                for metric in metrics:
                    try:
                        await self._insert_metric(metric)
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"Failed to store individual metric: {e}")
                        continue

                await self._connection.commit()
                self._total_operations += stored_count

            return stored_count

        except Exception as e:
            self._failed_operations += 1
            logger.error(f"Failed to store metrics batch: {e}")
            return stored_count

    async def get_metric(self, metric_id: UUID) -> MetricRecord | None:
        """Retrieve a single metric record by ID."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            async with self._connection.execute(
                "SELECT * FROM metrics WHERE id = ?", (str(metric_id),)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._row_to_metric(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get metric {metric_id}: {e}")
            return None

    async def get_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        order_desc: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[MetricRecord]:
        """Retrieve multiple metric records with filtering."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Build query
            query, params = self._build_select_query(
                start_time,
                end_time,
                metric_type,
                user_id,
                session_id,
                request_id,
                limit,
                offset,
                order_by,
                order_desc,
                filters,
            )

            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_metric(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []

    async def count_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count metric records matching the criteria."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Build count query
            query, params = self._build_count_query(
                start_time,
                end_time,
                metric_type,
                user_id,
                session_id,
                request_id,
                filters,
            )

            async with self._connection.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

        except Exception as e:
            logger.error(f"Failed to count metrics: {e}")
            return 0

    async def delete_metrics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        metric_type: MetricType | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Delete metric records matching the criteria."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Build delete query
            query, params = self._build_delete_query(
                start_time,
                end_time,
                metric_type,
                user_id,
                session_id,
                request_id,
                filters,
            )

            async with self._lock:
                cursor = await self._connection.execute(query, params)
                deleted_count = cursor.rowcount
                await self._connection.commit()

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete metrics: {e}")
            return 0

    async def get_metrics_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: str | None = None,
        session_id: str | None = None,
        group_by: str | None = None,
    ) -> MetricsSummary:
        """Get aggregated metrics summary."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Build summary query
            summary_query = self._build_summary_query(
                start_time, end_time, user_id, session_id
            )

            async with self._connection.execute(
                summary_query[0], summary_query[1]
            ) as cursor:
                row = await cursor.fetchone()

                if not row:
                    return MetricsSummary(start_time=start_time, end_time=end_time)

                # Extract summary data
                (
                    total_requests,
                    successful_requests,
                    failed_requests,
                    avg_response_time,
                    total_input_tokens,
                    total_output_tokens,
                    total_cost,
                    unique_users,
                ) = row

                summary = MetricsSummary(
                    start_time=start_time,
                    end_time=end_time,
                    total_requests=total_requests or 0,
                    successful_requests=successful_requests or 0,
                    failed_requests=failed_requests or 0,
                    avg_response_time_ms=avg_response_time or 0.0,
                    total_input_tokens=total_input_tokens or 0,
                    total_output_tokens=total_output_tokens or 0,
                    total_cost=total_cost or 0.0,
                    unique_users=unique_users or 0,
                )

                # Calculate derived metrics
                if summary.total_requests > 0:
                    summary.error_rate = (
                        summary.failed_requests / summary.total_requests
                    )
                    summary.avg_cost_per_request = (
                        summary.total_cost / summary.total_requests
                    )

                summary.total_tokens = (
                    summary.total_input_tokens + summary.total_output_tokens
                )

                return summary

        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return MetricsSummary(start_time=start_time, end_time=end_time)

    async def get_time_series(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1h",
        metric_type: MetricType | None = None,
        aggregation: str = "count",
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get time series data for metrics."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Build time series query
            query, params = self._build_time_series_query(
                start_time,
                end_time,
                interval,
                metric_type,
                aggregation,
                user_id,
                session_id,
            )

            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                return [
                    {
                        "timestamp": row[0],
                        "value": row[1],
                        "count": row[2] if len(row) > 2 else row[1],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get time series: {e}")
            return []

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Test connection
            async with self._connection.execute("SELECT 1") as cursor:
                await cursor.fetchone()

            # Get database info
            async with self._connection.execute(
                "SELECT COUNT(*) FROM metrics"
            ) as cursor:
                row = await cursor.fetchone()
                total_metrics = row[0] if row else 0

            return {
                "status": "healthy",
                "database_path": str(self.database_path),
                "total_metrics": total_metrics,
                "total_operations": self._total_operations,
                "failed_operations": self._failed_operations,
                "success_rate": (
                    (self._total_operations - self._failed_operations)
                    / self._total_operations
                    if self._total_operations > 0
                    else 1.0
                ),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_path": str(self.database_path),
            }

    async def get_storage_info(self) -> dict[str, Any]:
        """Get storage information."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            # Get database size
            db_size = (
                self.database_path.stat().st_size if self.database_path.exists() else 0
            )

            # Get table info
            async with self._connection.execute(
                "SELECT COUNT(*) FROM metrics"
            ) as cursor:
                row = await cursor.fetchone()
                total_metrics = row[0] if row else 0

            # Get metrics by type
            async with self._connection.execute(
                "SELECT metric_type, COUNT(*) FROM metrics GROUP BY metric_type"
            ) as cursor:
                type_counts = {row[0]: row[1] for row in await cursor.fetchall()}

            return {
                "backend": "sqlite",
                "database_path": str(self.database_path),
                "database_size_bytes": db_size,
                "total_metrics": total_metrics,
                "metrics_by_type": type_counts,
                "schema_version": self.SCHEMA_VERSION,
                "wal_enabled": self.enable_wal,
                "total_operations": self._total_operations,
                "failed_operations": self._failed_operations,
            }

        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {"backend": "sqlite", "error": str(e)}

    async def vacuum(self) -> None:
        """Perform database maintenance."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            async with self._lock:
                await self._connection.execute("VACUUM")
                await self._connection.execute("ANALYZE")
                await self._connection.commit()

            logger.info("Database vacuum and analyze completed")

        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")

    async def backup(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            backup_path_obj = Path(backup_path)
            backup_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # SQLite backup using the backup API
            async with aiosqlite.connect(str(backup_path_obj)) as backup_conn:
                await self._connection.backup(backup_conn)

            logger.info(f"Database backed up to {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    async def get_schema_version(self) -> str:
        """Get the current schema version."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            async with self._connection.execute(
                "SELECT version FROM schema_info ORDER BY created_at DESC LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else "unknown"

        except Exception:
            return "unknown"

    # Helper methods

    async def _ensure_schema_version(self) -> None:
        """Ensure schema version is recorded."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        try:
            async with self._connection.execute(
                "INSERT OR IGNORE INTO schema_info (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            ):
                pass
        except Exception as e:
            logger.warning(f"Failed to record schema version: {e}")

    async def _insert_metric(self, metric: MetricRecord) -> None:
        """Insert a metric record into the database."""
        self._ensure_connection()
        assert self._connection is not None  # guaranteed by _ensure_connection()
        # Convert metric to database row
        values = self._metric_to_values(metric)

        # Build insert query
        placeholders = ", ".join(["?" for _ in values])
        columns = ", ".join(
            [
                "id",
                "timestamp",
                "metric_type",
                "request_id",
                "user_id",
                "session_id",
                "metadata",
                "method",
                "path",
                "endpoint",
                "api_version",
                "client_ip",
                "user_agent",
                "content_length",
                "content_type",
                "model",
                "provider",
                "max_tokens",
                "temperature",
                "streaming",
                "status_code",
                "response_time_ms",
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "first_token_time_ms",
                "stream_completion_time_ms",
                "completion_reason",
                "safety_filtered",
                "error_type",
                "error_code",
                "error_message",
                "stack_trace",
                "retry_count",
                "recoverable",
                "input_cost",
                "output_cost",
                "cache_read_cost",
                "cache_write_cost",
                "total_cost",
                "pricing_tier",
                "currency",
                "request_processing_ms",
                "claude_api_call_ms",
                "response_processing_ms",
                "total_latency_ms",
                "queue_time_ms",
                "wait_time_ms",
                "first_token_latency_ms",
                "token_generation_rate",
                "request_count",
                "token_count",
                "window_start",
                "window_end",
                "window_duration_seconds",
                "aggregation_level",
            ]
        )

        query = f"INSERT INTO metrics ({columns}) VALUES ({placeholders})"

        await self._connection.execute(query, values)

    def _metric_to_values(self, metric: MetricRecord) -> tuple[Any, ...]:
        """Convert a metric record to database values."""
        # Base values
        values = [
            str(metric.id),
            metric.timestamp.isoformat(),
            metric.metric_type.value,
            metric.request_id,
            metric.user_id,
            metric.session_id,
            json.dumps(metric.metadata) if metric.metadata else None,
        ]

        # Initialize all field values to None
        field_values: list[Any] = [None] * 52  # Total number of optional fields

        # Fill in values based on metric type
        if isinstance(metric, RequestMetric):
            field_values[0:13] = [
                metric.method,
                metric.path,
                metric.endpoint,
                metric.api_version,
                metric.client_ip,
                metric.user_agent,
                metric.content_length,
                metric.content_type,
                metric.model,
                metric.provider,
                metric.max_tokens,
                metric.temperature,
                metric.streaming,
            ]
        elif isinstance(metric, ResponseMetric):
            field_values[13:23] = [
                metric.status_code,
                metric.response_time_ms,
                metric.input_tokens,
                metric.output_tokens,
                metric.cache_read_tokens,
                metric.cache_write_tokens,
                metric.first_token_time_ms,
                metric.stream_completion_time_ms,
                metric.completion_reason,
                metric.safety_filtered,
            ]
        elif isinstance(metric, ErrorMetric):
            field_values[23:29] = [
                metric.error_type,
                metric.error_code,
                metric.error_message,
                metric.stack_trace,
                metric.retry_count,
                metric.recoverable,
            ]
        elif isinstance(metric, CostMetric):
            field_values[29:36] = [
                metric.input_cost,
                metric.output_cost,
                metric.cache_read_cost,
                metric.cache_write_cost,
                metric.total_cost,
                metric.pricing_tier,
                metric.currency,
            ]
        elif isinstance(metric, LatencyMetric):
            field_values[36:44] = [
                metric.request_processing_ms,
                metric.claude_api_call_ms,
                metric.response_processing_ms,
                metric.total_latency_ms,
                metric.queue_time_ms,
                metric.wait_time_ms,
                metric.first_token_latency_ms,
                metric.token_generation_rate,
            ]
        elif isinstance(metric, UsageMetric):
            field_values[44:52] = [
                metric.request_count,
                metric.token_count,
                metric.window_start.isoformat(),
                metric.window_end.isoformat(),
                metric.window_duration_seconds,
                metric.aggregation_level,
                None,
                None,  # Padding
            ]

        return tuple(values + field_values)

    def _row_to_metric(self, row: sqlite3.Row) -> MetricRecord:
        """Convert a database row to a metric record."""
        # Extract base fields
        base_data = {
            "id": UUID(row["id"]),
            "timestamp": datetime.fromisoformat(row["timestamp"]),
            "request_id": row["request_id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

        metric_type = MetricType(row["metric_type"])

        # Create appropriate metric type based on type
        if metric_type == MetricType.REQUEST:
            return RequestMetric(
                **base_data,
                method=row["method"],
                path=row["path"],
                endpoint=row["endpoint"],
                api_version=row["api_version"],
                client_ip=row["client_ip"],
                user_agent=row["user_agent"],
                content_length=row["content_length"],
                content_type=row["content_type"],
                model=row["model"],
                provider=row["provider"],
                max_tokens=row["max_tokens"],
                temperature=row["temperature"],
                streaming=bool(row["streaming"])
                if row["streaming"] is not None
                else False,
            )
        elif metric_type == MetricType.RESPONSE:
            return ResponseMetric(
                **base_data,
                status_code=row["status_code"],
                response_time_ms=row["response_time_ms"],
                content_length=row["content_length"],
                content_type=row["content_type"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cache_read_tokens=row["cache_read_tokens"],
                cache_write_tokens=row["cache_write_tokens"],
                streaming=bool(row["streaming"])
                if row["streaming"] is not None
                else False,
                first_token_time_ms=row["first_token_time_ms"],
                stream_completion_time_ms=row["stream_completion_time_ms"],
                completion_reason=row["completion_reason"],
                safety_filtered=bool(row["safety_filtered"])
                if row["safety_filtered"] is not None
                else False,
            )
        elif metric_type == MetricType.ERROR:
            return ErrorMetric(
                **base_data,
                error_type=row["error_type"],
                error_code=row["error_code"],
                error_message=row["error_message"],
                stack_trace=row["stack_trace"],
                endpoint=row["endpoint"],
                method=row["method"],
                status_code=row["status_code"],
                retry_count=row["retry_count"] or 0,
                recoverable=bool(row["recoverable"])
                if row["recoverable"] is not None
                else False,
            )
        elif metric_type == MetricType.COST:
            return CostMetric(
                **base_data,
                input_cost=row["input_cost"] or 0.0,
                output_cost=row["output_cost"] or 0.0,
                cache_read_cost=row["cache_read_cost"] or 0.0,
                cache_write_cost=row["cache_write_cost"] or 0.0,
                total_cost=row["total_cost"] or 0.0,
                model=row["model"],
                pricing_tier=row["pricing_tier"],
                currency=row["currency"] or "USD",
                input_tokens=row["input_tokens"] or 0,
                output_tokens=row["output_tokens"] or 0,
                cache_read_tokens=row["cache_read_tokens"] or 0,
                cache_write_tokens=row["cache_write_tokens"] or 0,
            )
        elif metric_type == MetricType.LATENCY:
            return LatencyMetric(
                **base_data,
                request_processing_ms=row["request_processing_ms"] or 0.0,
                claude_api_call_ms=row["claude_api_call_ms"] or 0.0,
                response_processing_ms=row["response_processing_ms"] or 0.0,
                total_latency_ms=row["total_latency_ms"] or 0.0,
                queue_time_ms=row["queue_time_ms"] or 0.0,
                wait_time_ms=row["wait_time_ms"] or 0.0,
                first_token_latency_ms=row["first_token_latency_ms"],
                token_generation_rate=row["token_generation_rate"],
            )
        elif metric_type == MetricType.USAGE:
            return UsageMetric(
                **base_data,
                request_count=row["request_count"] or 1,
                token_count=row["token_count"] or 0,
                window_start=datetime.fromisoformat(row["window_start"]),
                window_end=datetime.fromisoformat(row["window_end"]),
                window_duration_seconds=row["window_duration_seconds"] or 0.0,
                aggregation_level=row["aggregation_level"] or "hourly",
            )
        else:
            # This should never happen since we handle all MetricType enum values above
            # If we reach here, it means a new MetricType was added but not handled
            raise ValueError(f"Unknown metric type: {metric_type}")

    def _build_select_query(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        metric_type: MetricType | None,
        user_id: str | None,
        session_id: str | None,
        request_id: str | None,
        limit: int | None,
        offset: int | None,
        order_by: str | None,
        order_desc: bool,
        filters: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build SELECT query with filters."""
        query = "SELECT * FROM metrics"
        params = []
        conditions = []

        # Add time filters
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp < ?")
            params.append(end_time.isoformat())

        # Add type filter
        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if request_id:
            conditions.append("request_id = ?")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)

        # Add WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add ORDER BY
        order_field = order_by or "timestamp"
        order_direction = "DESC" if order_desc else "ASC"
        query += f" ORDER BY {order_field} {order_direction}"

        # Add LIMIT and OFFSET
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        return query, params

    def _build_count_query(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        metric_type: MetricType | None,
        user_id: str | None,
        session_id: str | None,
        request_id: str | None,
        filters: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build COUNT query with filters."""
        query = "SELECT COUNT(*) FROM metrics"
        params = []
        conditions = []

        # Add time filters
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp < ?")
            params.append(end_time.isoformat())

        # Add type filter
        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if request_id:
            conditions.append("request_id = ?")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)

        # Add WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return query, params

    def _build_delete_query(
        self,
        start_time: datetime | None,
        end_time: datetime | None,
        metric_type: MetricType | None,
        user_id: str | None,
        session_id: str | None,
        request_id: str | None,
        filters: dict[str, Any] | None,
    ) -> tuple[str, list[Any]]:
        """Build DELETE query with filters."""
        query = "DELETE FROM metrics"
        params = []
        conditions = []

        # Add time filters
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        if end_time:
            conditions.append("timestamp < ?")
            params.append(end_time.isoformat())

        # Add type filter
        if metric_type:
            conditions.append("metric_type = ?")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if request_id:
            conditions.append("request_id = ?")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                conditions.append(f"{key} = ?")
                params.append(value)

        # Add WHERE clause
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        return query, params

    def _build_summary_query(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: str | None,
        session_id: str | None,
    ) -> tuple[str, list[Any]]:
        """Build summary aggregation query."""
        query = """
        SELECT
            SUM(CASE WHEN metric_type = 'request' THEN 1 ELSE 0 END) as total_requests,
            SUM(CASE WHEN metric_type = 'response' AND status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as successful_requests,
            SUM(CASE WHEN metric_type = 'response' AND status_code >= 400 THEN 1 ELSE 0 END) as failed_requests,
            AVG(CASE WHEN metric_type = 'response' THEN response_time_ms END) as avg_response_time,
            SUM(CASE WHEN metric_type = 'response' THEN COALESCE(input_tokens, 0) ELSE 0 END) as total_input_tokens,
            SUM(CASE WHEN metric_type = 'response' THEN COALESCE(output_tokens, 0) ELSE 0 END) as total_output_tokens,
            SUM(CASE WHEN metric_type = 'cost' THEN COALESCE(total_cost, 0) ELSE 0 END) as total_cost,
            COUNT(DISTINCT user_id) as unique_users
        FROM metrics
        WHERE timestamp >= ? AND timestamp < ?
        """

        params = [start_time.isoformat(), end_time.isoformat()]

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        return query, params

    def _build_time_series_query(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str,
        metric_type: MetricType | None,
        aggregation: str,
        user_id: str | None,
        session_id: str | None,
    ) -> tuple[str, list[Any]]:
        """Build time series query."""
        # Simple time series implementation
        # For more advanced time series, consider using SQLite's datetime functions

        if aggregation == "count":
            select_clause = "COUNT(*)"
        elif aggregation == "sum":
            select_clause = "SUM(COALESCE(total_cost, 1))"  # Default sum field
        elif aggregation == "avg":
            select_clause = "AVG(COALESCE(response_time_ms, 0))"  # Default avg field
        else:
            select_clause = "COUNT(*)"

        # Use simple bucketing for now
        query = f"""
        SELECT
            datetime(timestamp, 'start of hour') as time_bucket,
            {select_clause} as value,
            COUNT(*) as count
        FROM metrics
        WHERE timestamp >= ? AND timestamp < ?
        """

        params = [start_time.isoformat(), end_time.isoformat()]

        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type.value)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " GROUP BY time_bucket ORDER BY time_bucket"

        return query, params
