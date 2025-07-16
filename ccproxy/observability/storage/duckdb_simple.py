"""Simplified DuckDB storage for low-traffic environments.

This module provides a simple, direct DuckDB storage implementation without
connection pooling or batch processing. Suitable for dev environments with
low request rates (< 10 req/s).
"""

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


class SimpleDuckDBStorage:
    """Simple DuckDB storage without pooling or batching."""

    def __init__(self, database_path: str | Path = "data/metrics.duckdb"):
        """Initialize simple DuckDB storage.

        Args:
            database_path: Path to DuckDB database file
        """
        if not DUCKDB_AVAILABLE:
            logger.warning("duckdb_not_available", install_cmd="pip install duckdb")

        self.database_path = Path(database_path)
        self._connection: Any | None = None
        self._initialized = False
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

            # Create connection
            self._connection = duckdb.connect(str(self.database_path))

            # Create schema
            await self._create_schema()

            self._initialized = True
            logger.debug(
                "simple_duckdb_initialized", database_path=str(self.database_path)
            )

        except Exception as e:
            logger.error("simple_duckdb_init_error", error=str(e), exc_info=True)
            self._enabled = False

    async def _create_schema(self) -> None:
        """Create database schema for metrics."""
        if not self._connection:
            return

        try:
            # Main requests table - same as before
            self._connection.execute("""
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

            # Create basic indexes
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_requests_request_id ON requests(request_id)"
            )

        except Exception as e:
            logger.error("simple_duckdb_schema_error", error=str(e))

    async def store_request(self, data: dict[str, Any]) -> bool:
        """Store a single request log entry.

        Args:
            data: Request data to store

        Returns:
            True if stored successfully
        """
        if not self._enabled or not self._initialized or not self._connection:
            return False

        try:
            # Prepare data for insert
            insert_data = (
                data.get("timestamp", time.time()),
                data.get("request_id"),
                data.get("method"),
                data.get("endpoint"),
                data.get("service_type"),
                data.get("model"),
                data.get("status"),
                data.get("response_time", 0.0),
                data.get("tokens_input", 0),
                data.get("tokens_output", 0),
                data.get("cost_usd", 0.0),
                data.get("error_type"),
                data.get("error_message"),
                json.dumps(data.get("metadata", {})),
            )

            # Direct insert with explicit column names
            self._connection.execute(
                """
                INSERT INTO requests (
                    timestamp, request_id, method, endpoint, service_type,
                    model, status, response_time, tokens_input, tokens_output,
                    cost_usd, error_type, error_message, metadata
                ) VALUES (
                    to_timestamp(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                insert_data,
            )

            return True

        except Exception as e:
            logger.error(
                "simple_duckdb_store_error",
                error=str(e),
                request_id=data.get("request_id"),
            )
            return False

    async def query(
        self,
        sql: str,
        params: list[Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Execute SQL query and return results.

        Args:
            sql: SQL query string
            params: Query parameters
            limit: Maximum number of results

        Returns:
            List of result rows as dictionaries
        """
        if not self._enabled or not self._initialized or not self._connection:
            return []

        try:
            # Apply limit to query
            limited_sql = f"SELECT * FROM ({sql}) LIMIT {limit}"

            # Execute query
            if params:
                result = self._connection.execute(limited_sql, params)
            else:
                result = self._connection.execute(limited_sql)

            # Convert to list of dictionaries
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()

            return [dict(zip(columns, row, strict=False)) for row in rows]

        except Exception as e:
            logger.error("simple_duckdb_query_error", sql=sql, error=str(e))
            return []

    async def get_recent_requests(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent requests for debugging/monitoring.

        Args:
            limit: Number of recent requests to return

        Returns:
            List of recent request records
        """
        return await self.query(
            "SELECT * FROM requests ORDER BY timestamp DESC", limit=limit
        )

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.error("simple_duckdb_close_error", error=str(e))
            finally:
                self._connection = None
                self._initialized = False

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
            if not self._connection:
                return {
                    "status": "no_connection",
                    "enabled": False,
                }

            # Test query
            result = self._connection.execute("SELECT COUNT(*) FROM requests")
            request_count = result.fetchone()[0]

            return {
                "status": "healthy",
                "enabled": True,
                "database_path": str(self.database_path),
                "request_count": request_count,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "enabled": False,
                "error": str(e),
            }
