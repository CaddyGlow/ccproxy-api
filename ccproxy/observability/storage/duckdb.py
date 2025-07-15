"""
DuckDB storage backend for analytics-optimized metrics storage.

This storage backend uses DuckDB for excellent analytical query performance
on structured log events and metrics data. DuckDB provides:
- Fast analytical queries (OLAP workload)
- Excellent compression for time-series data
- Full SQL support with advanced analytics functions
- Zero configuration - single file database
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import structlog


# Handle graceful degradation if DuckDB not available
try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    duckdb = None  # type: ignore[assignment]


logger = structlog.get_logger(__name__)


class DuckDBStorage:
    """
    DuckDB storage backend for metrics analytics.

    Provides async interface for storing and querying metrics data
    with excellent analytical performance.
    """

    def __init__(
        self,
        database_path: str | Path = "data/metrics.duckdb",
        pool_size: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize DuckDB storage.

        Args:
            database_path: Path to DuckDB database file
            pool_size: Number of connections in pool
            timeout: Query timeout in seconds
        """
        if not DUCKDB_AVAILABLE:
            logger.warning("duckdb_not_available", install_cmd="pip install duckdb")

        self.database_path = Path(database_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self._initialized = False
        self._connection_pool: list[Any] = []
        self._pool_lock = asyncio.Lock()
        self._enabled = DUCKDB_AVAILABLE

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        if not self._enabled:
            logger.info("duckdb_storage_disabled", reason="duckdb_not_available")
            return

        if self._initialized:
            return

        try:
            # Ensure data directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize connection pool
            await self._init_connection_pool()

            # Create tables if they don't exist
            await self._create_schema()

            self._initialized = True
            logger.info(
                "duckdb_storage_initialized",
                database_path=str(self.database_path),
                pool_size=self.pool_size,
            )

        except Exception as e:
            logger.error("duckdb_init_error", error=str(e), exc_info=True)
            self._enabled = False

    async def _init_connection_pool(self) -> None:
        """Initialize connection pool."""
        if not self._enabled:
            return

        async with self._pool_lock:
            # Create connections in pool
            for _ in range(self.pool_size):
                conn = duckdb.connect(str(self.database_path))
                # Configure for analytics workload
                conn.execute("PRAGMA threads=4")
                conn.execute("PRAGMA memory_limit='512MB'")
                self._connection_pool.append(conn)

    async def _get_connection(self) -> Any:
        """Get connection from pool."""
        if not self._enabled:
            return None

        async with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()

        # If pool empty, create temporary connection
        if duckdb:
            return duckdb.connect(str(self.database_path))
        return None

    async def _return_connection(self, conn: Any) -> None:
        """Return connection to pool."""
        if not self._enabled or not conn:
            return

        async with self._pool_lock:
            if len(self._connection_pool) < self.pool_size:
                self._connection_pool.append(conn)
            else:
                conn.close()

    async def _create_schema(self) -> None:
        """Create database schema for metrics."""
        if not self._enabled:
            return

        conn = await self._get_connection()
        if not conn:
            return

        try:
            # Main requests table for API requests
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    timestamp TIMESTAMP,
                    request_id VARCHAR,
                    method VARCHAR,
                    endpoint VARCHAR,
                    service_type VARCHAR,
                    model VARCHAR,
                    status VARCHAR,
                    response_time DOUBLE,
                    tokens_input INTEGER,
                    tokens_output INTEGER,
                    cost_usd DOUBLE,
                    error_type VARCHAR,
                    error_message TEXT,
                    metadata JSON
                )
            """)

            # Operations table for timed operations within requests
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    timestamp TIMESTAMP,
                    request_id VARCHAR,
                    operation_id VARCHAR,
                    operation_name VARCHAR,
                    duration_ms DOUBLE,
                    status VARCHAR,
                    error_type VARCHAR,
                    metadata JSON
                )
            """)

            # Add service_type column if it doesn't exist (migration for existing databases)
            try:
                conn.execute("ALTER TABLE requests ADD COLUMN service_type VARCHAR")
                logger.info(
                    "duckdb_schema_migration", action="added_service_type_column"
                )
            except Exception:
                # Column already exists or other error, continue
                pass

            # Create indexes for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_endpoint ON requests(endpoint)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_service_type ON requests(service_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON operations(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_operations_request_id ON operations(request_id)"
            )


        except Exception as e:
            logger.error("duckdb_schema_error", error=str(e))
        finally:
            await self._return_connection(conn)

    async def store(self, metric: dict[str, Any]) -> bool:
        """
        Store single metric.

        Args:
            metric: Metric data to store

        Returns:
            True if stored successfully
        """
        return await self.store_batch([metric])

    async def store_batch(self, metrics: list[dict[str, Any]]) -> bool:
        """
        Store batch of metrics efficiently.

        Args:
            metrics: List of metric data to store

        Returns:
            True if batch stored successfully
        """
        if not self._enabled or not self._initialized or not metrics:
            return False

        conn = await self._get_connection()
        if not conn:
            return False

        try:
            # Separate requests and operations
            requests = []
            operations = []

            for metric in metrics:
                if metric.get("operation_id"):
                    operations.append(metric)
                else:
                    requests.append(metric)

            # Insert requests
            if requests:
                await self._insert_requests(conn, requests)

            # Insert operations
            if operations:
                await self._insert_operations(conn, operations)

            return True

        except Exception as e:
            logger.error("duckdb_store_error", error=str(e), metric_count=len(metrics))
            return False
        finally:
            await self._return_connection(conn)

    async def _insert_requests(self, conn: Any, requests: list[dict[str, Any]]) -> None:
        """Insert request metrics."""
        if not requests:
            return

        # Prepare data for batch insert
        data = []
        for req in requests:
            data.append(
                (
                    req.get("timestamp", time.time()),
                    req.get("request_id"),
                    req.get("method"),
                    req.get("endpoint"),
                    req.get("service_type"),
                    req.get("model"),
                    req.get("status"),
                    req.get("response_time", 0.0),
                    req.get("tokens_input", 0),
                    req.get("tokens_output", 0),
                    req.get("cost_usd", 0.0),
                    req.get("error_type"),
                    req.get("error_message"),
                    json.dumps(req.get("metadata", {})),
                )
            )

        # Batch insert with timestamp conversion using explicit column names
        conn.executemany(
            """
            INSERT INTO requests (timestamp, request_id, method, endpoint, service_type, model, status, response_time, tokens_input, tokens_output, cost_usd, error_type, error_message, metadata)
            VALUES (to_timestamp(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            data,
        )

    async def _insert_operations(
        self, conn: Any, operations: list[dict[str, Any]]
    ) -> None:
        """Insert operation metrics."""
        if not operations:
            return

        # Prepare data for batch insert
        data = []
        for op in operations:
            data.append(
                (
                    op.get("timestamp", time.time()),
                    op.get("request_id"),
                    op.get("operation_id"),
                    op.get("operation_name"),
                    op.get("duration_ms", 0.0),
                    op.get("status"),
                    op.get("error_type"),
                    json.dumps(op.get("metadata", {})),
                )
            )

        # Batch insert with timestamp conversion
        conn.executemany(
            """
            INSERT INTO operations VALUES (to_timestamp(?), ?, ?, ?, ?, ?, ?, ?)
        """,
            data,
        )

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Execute SQL query and return results.

        Args:
            sql: SQL query string
            params: Query parameters
            limit: Maximum number of results

        Returns:
            List of result rows as dictionaries
        """
        if not self._enabled or not self._initialized:
            return []

        conn = await self._get_connection()
        if not conn:
            return []

        try:
            # Apply limit to query
            limited_sql = f"SELECT * FROM ({sql}) LIMIT {limit}"

            # Execute query
            if params:
                result = conn.execute(limited_sql, params)
            else:
                result = conn.execute(limited_sql)

            # Convert to list of dictionaries
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            return [dict(zip(columns, row, strict=False)) for row in rows]

        except Exception as e:
            logger.error("duckdb_query_error", sql=sql, error=str(e))
            return []
        finally:
            await self._return_connection(conn)

    async def get_analytics(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        model: str | None = None,
        service_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get analytics summary for the specified time range.

        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            model: Filter by model name
            service_type: Filter by service type (proxy_service or claude_sdk_service)

        Returns:
            Analytics summary data
        """
        if not self._enabled or not self._initialized:
            return {}

        conn = await self._get_connection()
        if not conn:
            return {}

        try:
            # Build WHERE clause
            where_conditions = []
            params: list[Any] = []

            if start_time:
                where_conditions.append("timestamp >= to_timestamp(?)")
                params.append(start_time)

            if end_time:
                where_conditions.append("timestamp <= to_timestamp(?)")
                params.append(end_time)

            if model:
                where_conditions.append("model = ?")
                params.append(model)

            if service_type:
                where_conditions.append("service_type = ?")
                params.append(service_type)

            where_clause = (
                "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            )

            # Get summary statistics
            summary_sql = f"""
                SELECT
                    COUNT(*) as total_requests,
                    COUNT(*) FILTER (WHERE status = 'success') as successful_requests,
                    COUNT(*) FILTER (WHERE status = 'error') as failed_requests,
                    AVG(response_time) as avg_response_time,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as median_response_time,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time) as p95_response_time,
                    SUM(tokens_input) as total_tokens_input,
                    SUM(tokens_output) as total_tokens_output,
                    SUM(cost_usd) as total_cost_usd
                FROM requests {where_clause}
            """

            result = conn.execute(summary_sql, params)
            summary = dict(
                zip(
                    [desc[0] for desc in result.description],
                    result.fetchone(),
                    strict=False,
                )
            )

            # Get hourly request counts
            hourly_sql = f"""
                SELECT
                    date_trunc('hour', timestamp) as hour,
                    COUNT(*) as request_count,
                    COUNT(*) FILTER (WHERE status = 'error') as error_count
                FROM requests {where_clause}
                GROUP BY hour
                ORDER BY hour
                LIMIT 168
            """

            result = conn.execute(hourly_sql, params)
            hourly_data = [
                dict(zip([desc[0] for desc in result.description], row, strict=False))
                for row in result.fetchall()
            ]

            # Get model statistics
            model_sql = f"""
                SELECT
                    model,
                    COUNT(*) as request_count,
                    AVG(response_time) as avg_response_time,
                    SUM(cost_usd) as total_cost
                FROM requests {where_clause}
                GROUP BY model
                ORDER BY request_count DESC
                LIMIT 20
            """

            result = conn.execute(model_sql, params)
            model_stats = [
                dict(zip([desc[0] for desc in result.description], row, strict=False))
                for row in result.fetchall()
            ]

            # Get service type breakdown if not filtering by service_type
            service_breakdown = []
            if not service_type:
                service_sql = f"""
                    SELECT
                        COALESCE(service_type, 'unknown') as service_type,
                        COUNT(*) as request_count,
                        AVG(response_time) as avg_response_time,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time) as median_response_time,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time) as p95_response_time,
                        SUM(cost_usd) as total_cost
                    FROM requests {where_clause}
                    GROUP BY service_type
                    ORDER BY request_count DESC
                """

                result = conn.execute(service_sql, params)
                service_breakdown = [
                    dict(
                        zip([desc[0] for desc in result.description], row, strict=False)
                    )
                    for row in result.fetchall()
                ]

            return {
                "summary": summary,
                "hourly_data": hourly_data,
                "model_stats": model_stats,
                "service_breakdown": service_breakdown,
                "query_time": time.time(),
            }

        except Exception as e:
            logger.error("duckdb_analytics_error", error=str(e))
            return {}
        finally:
            await self._return_connection(conn)

    async def close(self) -> None:
        """Close all connections and cleanup."""
        if not self._enabled:
            return

        from contextlib import suppress

        async with self._pool_lock:
            for conn in self._connection_pool:
                with suppress(Exception):
                    conn.close()
            self._connection_pool.clear()

        self._initialized = False
        logger.info("duckdb_storage_closed")

    def is_enabled(self) -> bool:
        """Check if storage is enabled and available."""
        return self._enabled and self._initialized

    async def health_check(self) -> dict[str, Any]:
        """Get health status of the storage backend."""
        if not self._enabled:
            return {
                "status": "disabled",
                "reason": "duckdb_not_available",
                "enabled": False,
            }

        if not self._initialized:
            return {
                "status": "not_initialized",
                "enabled": False,
            }

        try:
            conn = await self._get_connection()
            if not conn:
                return {
                    "status": "connection_failed",
                    "enabled": False,
                }

            # Test query
            result = conn.execute("SELECT COUNT(*) FROM requests")
            request_count = result.fetchone()[0]

            await self._return_connection(conn)

            return {
                "status": "healthy",
                "enabled": True,
                "database_path": str(self.database_path),
                "request_count": request_count,
                "pool_size": len(self._connection_pool),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "enabled": False,
                "error": str(e),
            }
