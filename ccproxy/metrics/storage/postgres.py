"""
PostgreSQL storage implementation for metrics.

This module provides a PostgreSQL-based storage backend for metrics,
suitable for high-scale production deployments with advanced analytics needs.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID


try:
    import asyncpg

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

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
    StorageError,
    StorageInitializationError,
    StorageOperationError,
)


logger = logging.getLogger(__name__)


class PostgreSQLMetricsStorage(MetricsStorage):
    """
    PostgreSQL storage implementation for metrics.

    This storage backend uses PostgreSQL for scalable persistent storage
    with advanced analytics capabilities.

    Requires asyncpg to be installed: pip install asyncpg
    """

    SCHEMA_VERSION = "1.0.0"

    # SQL schema
    CREATE_SCHEMA_SQL = """
    CREATE SCHEMA IF NOT EXISTS metrics;

    CREATE TABLE IF NOT EXISTS metrics.schema_info (
        version TEXT PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS metrics.metrics (
        id UUID PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
        metric_type TEXT NOT NULL,
        request_id TEXT,
        user_id TEXT,
        session_id TEXT,
        metadata JSONB,

        -- Request fields
        method TEXT,
        path TEXT,
        endpoint TEXT,
        api_version TEXT,
        client_ip INET,
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
        input_cost DECIMAL(10,6),
        output_cost DECIMAL(10,6),
        cache_read_cost DECIMAL(10,6),
        cache_write_cost DECIMAL(10,6),
        total_cost DECIMAL(10,6),
        pricing_tier TEXT,
        currency TEXT DEFAULT 'USD',

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
        window_start TIMESTAMP WITH TIME ZONE,
        window_end TIMESTAMP WITH TIME ZONE,
        window_duration_seconds REAL,
        aggregation_level TEXT,

        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics.metrics USING BTREE (timestamp);
    CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics.metrics USING BTREE (metric_type);
    CREATE INDEX IF NOT EXISTS idx_metrics_user_id ON metrics.metrics USING BTREE (user_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_session_id ON metrics.metrics USING BTREE (session_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_request_id ON metrics.metrics USING BTREE (request_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_model ON metrics.metrics USING BTREE (model);
    CREATE INDEX IF NOT EXISTS idx_metrics_provider ON metrics.metrics USING BTREE (provider);
    CREATE INDEX IF NOT EXISTS idx_metrics_composite ON metrics.metrics USING BTREE (timestamp, metric_type, user_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_metadata ON metrics.metrics USING GIN (metadata);
    CREATE INDEX IF NOT EXISTS idx_metrics_client_ip ON metrics.metrics USING BTREE (client_ip);

    -- Partitioning setup (optional, for very high volume)
    -- This would require manual setup based on specific requirements
    """

    def __init__(
        self,
        dsn: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "metrics",
        user: str = "postgres",
        password: str | None = None,
        ssl: str | None = None,
        pool_min_size: int = 10,
        pool_max_size: int = 20,
        command_timeout: float = 60.0,
    ):
        """
        Initialize the PostgreSQL storage.

        Args:
            dsn: Database connection string (if provided, overrides other connection params)
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            ssl: SSL mode ('require', 'prefer', 'disable')
            pool_min_size: Minimum connection pool size
            pool_max_size: Maximum connection pool size
            command_timeout: Command timeout in seconds
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError(
                "asyncpg is required for PostgreSQL storage. "
                "Install it with: pip install asyncpg"
            )

        self.dsn = dsn
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.ssl = ssl
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.command_timeout = command_timeout

        # Connection management
        self._pool: asyncpg.Pool | None = None

        # Performance tracking
        self._total_operations = 0
        self._failed_operations = 0

    async def initialize(self) -> None:
        """Initialize the PostgreSQL storage."""
        try:
            # Create connection pool
            if self.dsn:
                self._pool = await asyncpg.create_pool(
                    self.dsn,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    command_timeout=self.command_timeout,
                )
            else:
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    ssl=self.ssl,
                    min_size=self.pool_min_size,
                    max_size=self.pool_max_size,
                    command_timeout=self.command_timeout,
                )

            # Create schema and tables
            async with self._pool.acquire() as conn:
                await conn.execute(self.CREATE_SCHEMA_SQL)
                await self._ensure_schema_version(conn)

            logger.info("Initialized PostgreSQL metrics storage")

        except Exception as e:
            error_msg = f"Failed to initialize PostgreSQL storage: {e}"
            logger.error(error_msg)
            raise StorageInitializationError(error_msg) from e

    async def close(self) -> None:
        """Close the PostgreSQL storage."""
        if self._pool:
            await self._pool.close()
            self._pool = None

        logger.info("Closed PostgreSQL metrics storage")

    async def store_metric(self, metric: MetricRecord) -> bool:
        """Store a single metric record."""
        try:
            async with self._pool.acquire() as conn:
                await self._insert_metric(conn, metric)
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

        stored_count = 0

        try:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    for metric in metrics:
                        try:
                            await self._insert_metric(conn, metric)
                            stored_count += 1
                        except Exception as e:
                            logger.error(f"Failed to store individual metric: {e}")
                            continue

                self._total_operations += stored_count

            return stored_count

        except Exception as e:
            self._failed_operations += 1
            logger.error(f"Failed to store metrics batch: {e}")
            return stored_count

    async def get_metric(self, metric_id: UUID) -> MetricRecord | None:
        """Retrieve a single metric record by ID."""
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM metrics.metrics WHERE id = $1", metric_id
                )
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

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
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

            async with self._pool.acquire() as conn:
                result = await conn.fetchval(query, *params)
                return result or 0

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

            async with self._pool.acquire() as conn:
                result = await conn.execute(query, *params)
                # Extract number from "DELETE N" result
                deleted_count = int(result.split()[-1]) if result else 0

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
        try:
            # Build summary query
            query, params = self._build_summary_query(
                start_time, end_time, user_id, session_id
            )

            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)

                if not row:
                    return MetricsSummary(start_time=start_time, end_time=end_time)

                # Extract summary data
                summary = MetricsSummary(
                    start_time=start_time,
                    end_time=end_time,
                    total_requests=row["total_requests"] or 0,
                    successful_requests=row["successful_requests"] or 0,
                    failed_requests=row["failed_requests"] or 0,
                    avg_response_time_ms=float(row["avg_response_time"])
                    if row["avg_response_time"]
                    else 0.0,
                    total_input_tokens=row["total_input_tokens"] or 0,
                    total_output_tokens=row["total_output_tokens"] or 0,
                    total_cost=float(row["total_cost"]) if row["total_cost"] else 0.0,
                    unique_users=row["unique_users"] or 0,
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

            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                return [
                    {
                        "timestamp": row["time_bucket"].isoformat(),
                        "value": float(row["value"])
                        if row["value"] is not None
                        else 0.0,
                        "count": row["count"] or 0,
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get time series: {e}")
            return []

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            # Test connection
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")

            # Get database info
            async with self._pool.acquire() as conn:
                total_metrics = await conn.fetchval(
                    "SELECT COUNT(*) FROM metrics.metrics"
                )

            return {
                "status": "healthy",
                "database": self.database,
                "host": self.host,
                "port": self.port,
                "total_metrics": total_metrics or 0,
                "pool_size": self._pool.get_size(),
                "pool_idle": self._pool.get_idle_size(),
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
                "database": self.database,
                "host": self.host,
                "port": self.port,
            }

    async def get_storage_info(self) -> dict[str, Any]:
        """Get storage information."""
        try:
            async with self._pool.acquire() as conn:
                # Get table info
                total_metrics = await conn.fetchval(
                    "SELECT COUNT(*) FROM metrics.metrics"
                )

                # Get database size
                db_size = await conn.fetchval(
                    "SELECT pg_database_size($1)", self.database
                )

                # Get metrics by type
                type_counts_rows = await conn.fetch(
                    "SELECT metric_type, COUNT(*) FROM metrics.metrics GROUP BY metric_type"
                )
                type_counts = {
                    row["metric_type"]: row["count"] for row in type_counts_rows
                }

                return {
                    "backend": "postgresql",
                    "database": self.database,
                    "host": self.host,
                    "port": self.port,
                    "database_size_bytes": db_size or 0,
                    "total_metrics": total_metrics or 0,
                    "metrics_by_type": type_counts,
                    "schema_version": self.SCHEMA_VERSION,
                    "pool_size": self._pool.get_size(),
                    "pool_idle": self._pool.get_idle_size(),
                    "total_operations": self._total_operations,
                    "failed_operations": self._failed_operations,
                }

        except Exception as e:
            logger.error(f"Failed to get storage info: {e}")
            return {"backend": "postgresql", "error": str(e)}

    async def vacuum(self) -> None:
        """Perform database maintenance."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("VACUUM ANALYZE metrics.metrics")

            logger.info("Database vacuum and analyze completed")

        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")

    async def backup(self, backup_path: str) -> bool:
        """Create a backup using pg_dump."""
        try:
            import subprocess

            # Use pg_dump to create backup
            cmd = [
                "pg_dump",
                f"--host={self.host}",
                f"--port={self.port}",
                f"--username={self.user}",
                f"--dbname={self.database}",
                f"--file={backup_path}",
                "--verbose",
                "--schema=metrics",
            ]

            if self.password:
                # Set PGPASSWORD environment variable
                import os

                env = os.environ.copy()
                env["PGPASSWORD"] = self.password

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Database backed up to {backup_path}")
                return True
            else:
                logger.error(f"Backup failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            return False

    async def get_schema_version(self) -> str:
        """Get the current schema version."""
        try:
            async with self._pool.acquire() as conn:
                version = await conn.fetchval(
                    "SELECT version FROM metrics.schema_info ORDER BY created_at DESC LIMIT 1"
                )
                return version or "unknown"

        except Exception:
            return "unknown"

    # Helper methods

    async def _ensure_schema_version(self, conn: asyncpg.Connection) -> None:
        """Ensure schema version is recorded."""
        try:
            await conn.execute(
                "INSERT INTO metrics.schema_info (version) VALUES ($1) ON CONFLICT DO NOTHING",
                self.SCHEMA_VERSION,
            )
        except Exception as e:
            logger.warning(f"Failed to record schema version: {e}")

    async def _insert_metric(
        self, conn: asyncpg.Connection, metric: MetricRecord
    ) -> None:
        """Insert a metric record into the database."""
        # Convert metric to values
        values = self._metric_to_values(metric)

        # Build insert query
        query = """
        INSERT INTO metrics.metrics (
            id, timestamp, metric_type, request_id, user_id, session_id, metadata,
            method, path, endpoint, api_version, client_ip, user_agent,
            content_length, content_type, model, provider, max_tokens,
            temperature, streaming, status_code, response_time_ms,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            first_token_time_ms, stream_completion_time_ms, completion_reason,
            safety_filtered, error_type, error_code, error_message,
            stack_trace, retry_count, recoverable, input_cost, output_cost,
            cache_read_cost, cache_write_cost, total_cost, pricing_tier,
            currency, request_processing_ms, claude_api_call_ms,
            response_processing_ms, total_latency_ms, queue_time_ms,
            wait_time_ms, first_token_latency_ms, token_generation_rate,
            request_count, token_count, window_start, window_end,
            window_duration_seconds, aggregation_level
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
            $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
            $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41,
            $42, $43, $44, $45, $46, $47, $48, $49, $50, $51, $52, $53, $54,
            $55, $56, $57, $58, $59
        )
        """

        await conn.execute(query, *values)

    def _metric_to_values(self, metric: MetricRecord) -> tuple:
        """Convert a metric record to database values."""
        # Base values
        values = [
            metric.id,
            metric.timestamp,
            metric.metric_type.value,
            metric.request_id,
            metric.user_id,
            metric.session_id,
            json.dumps(metric.metadata) if metric.metadata else None,
        ]

        # Initialize all field values to None
        field_values = [None] * 52  # Total number of optional fields

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
                metric.window_start,
                metric.window_end,
                metric.window_duration_seconds,
                metric.aggregation_level,
                None,
                None,  # Padding
            ]

        return tuple(values + field_values)

    def _row_to_metric(self, row: asyncpg.Record) -> MetricRecord:
        """Convert a database row to a metric record."""
        # Extract base fields
        base_data = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "request_id": row["request_id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "metadata": row["metadata"] or {},
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
                client_ip=str(row["client_ip"]) if row["client_ip"] else None,
                user_agent=row["user_agent"],
                content_length=row["content_length"],
                content_type=row["content_type"],
                model=row["model"],
                provider=row["provider"],
                max_tokens=row["max_tokens"],
                temperature=row["temperature"],
                streaming=row["streaming"] or False,
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
                streaming=row["streaming"] or False,
                first_token_time_ms=row["first_token_time_ms"],
                stream_completion_time_ms=row["stream_completion_time_ms"],
                completion_reason=row["completion_reason"],
                safety_filtered=row["safety_filtered"] or False,
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
                recoverable=row["recoverable"] or False,
            )
        elif metric_type == MetricType.COST:
            return CostMetric(
                **base_data,
                input_cost=float(row["input_cost"]) if row["input_cost"] else 0.0,
                output_cost=float(row["output_cost"]) if row["output_cost"] else 0.0,
                cache_read_cost=float(row["cache_read_cost"])
                if row["cache_read_cost"]
                else 0.0,
                cache_write_cost=float(row["cache_write_cost"])
                if row["cache_write_cost"]
                else 0.0,
                total_cost=float(row["total_cost"]) if row["total_cost"] else 0.0,
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
                window_start=row["window_start"],
                window_end=row["window_end"],
                window_duration_seconds=row["window_duration_seconds"] or 0.0,
                aggregation_level=row["aggregation_level"] or "hourly",
            )
        else:
            # Return base metric record for unknown types
            return MetricRecord(
                **base_data,
                metric_type=metric_type,
            )

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
    ) -> tuple[str, list]:
        """Build SELECT query with filters."""
        query = "SELECT * FROM metrics.metrics"
        params = []
        conditions = []
        param_count = 0

        # Add time filters
        if start_time:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
        if end_time:
            param_count += 1
            conditions.append(f"timestamp < ${param_count}")
            params.append(end_time)

        # Add type filter
        if metric_type:
            param_count += 1
            conditions.append(f"metric_type = ${param_count}")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        if session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(session_id)
        if request_id:
            param_count += 1
            conditions.append(f"request_id = ${param_count}")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                param_count += 1
                conditions.append(f"{key} = ${param_count}")
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
    ) -> tuple[str, list]:
        """Build COUNT query with filters."""
        query = "SELECT COUNT(*) FROM metrics.metrics"
        params = []
        conditions = []
        param_count = 0

        # Add time filters
        if start_time:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
        if end_time:
            param_count += 1
            conditions.append(f"timestamp < ${param_count}")
            params.append(end_time)

        # Add type filter
        if metric_type:
            param_count += 1
            conditions.append(f"metric_type = ${param_count}")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        if session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(session_id)
        if request_id:
            param_count += 1
            conditions.append(f"request_id = ${param_count}")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                param_count += 1
                conditions.append(f"{key} = ${param_count}")
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
    ) -> tuple[str, list]:
        """Build DELETE query with filters."""
        query = "DELETE FROM metrics.metrics"
        params = []
        conditions = []
        param_count = 0

        # Add time filters
        if start_time:
            param_count += 1
            conditions.append(f"timestamp >= ${param_count}")
            params.append(start_time)
        if end_time:
            param_count += 1
            conditions.append(f"timestamp < ${param_count}")
            params.append(end_time)

        # Add type filter
        if metric_type:
            param_count += 1
            conditions.append(f"metric_type = ${param_count}")
            params.append(metric_type.value)

        # Add ID filters
        if user_id:
            param_count += 1
            conditions.append(f"user_id = ${param_count}")
            params.append(user_id)
        if session_id:
            param_count += 1
            conditions.append(f"session_id = ${param_count}")
            params.append(session_id)
        if request_id:
            param_count += 1
            conditions.append(f"request_id = ${param_count}")
            params.append(request_id)

        # Add custom filters
        if filters:
            for key, value in filters.items():
                param_count += 1
                conditions.append(f"{key} = ${param_count}")
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
    ) -> tuple[str, list]:
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
        FROM metrics.metrics
        WHERE timestamp >= $1 AND timestamp < $2
        """

        params = [start_time, end_time]
        param_count = 2

        if user_id:
            param_count += 1
            query += f" AND user_id = ${param_count}"
            params.append(user_id)
        if session_id:
            param_count += 1
            query += f" AND session_id = ${param_count}"
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
    ) -> tuple[str, list]:
        """Build time series query using PostgreSQL's date_trunc."""
        if aggregation == "count":
            select_clause = "COUNT(*)"
        elif aggregation == "sum":
            select_clause = "SUM(COALESCE(total_cost, 1))"  # Default sum field
        elif aggregation == "avg":
            select_clause = "AVG(COALESCE(response_time_ms, 0))"  # Default avg field
        else:
            select_clause = "COUNT(*)"

        # Map interval to PostgreSQL date_trunc format
        if interval.endswith("s"):
            pg_interval = "second"
        elif interval.endswith("m"):
            pg_interval = "minute"
        elif interval.endswith("h"):
            pg_interval = "hour"
        elif interval.endswith("d"):
            pg_interval = "day"
        else:
            pg_interval = "hour"

        query = f"""
        SELECT
            date_trunc('{pg_interval}', timestamp) as time_bucket,
            {select_clause} as value,
            COUNT(*) as count
        FROM metrics.metrics
        WHERE timestamp >= $1 AND timestamp < $2
        """

        params = [start_time, end_time]
        param_count = 2

        if metric_type:
            param_count += 1
            query += f" AND metric_type = ${param_count}"
            params.append(metric_type.value)
        if user_id:
            param_count += 1
            query += f" AND user_id = ${param_count}"
            params.append(user_id)
        if session_id:
            param_count += 1
            query += f" AND session_id = ${param_count}"
            params.append(session_id)

        query += " GROUP BY time_bucket ORDER BY time_bucket"

        return query, params
