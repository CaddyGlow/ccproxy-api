"""Simplified DuckDB storage for low-traffic environments.

This module provides a simple, direct DuckDB storage implementation without
connection pooling or batch processing. Suitable for dev environments with
low request rates (< 10 req/s).
"""

import asyncio
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine, desc, func, select
from typing_extensions import TypedDict

from .models import AccessLog


logger = structlog.get_logger(__name__)


class AccessLogPayload(TypedDict, total=False):
    """TypedDict for access log data payloads.

    Note: All fields are optional (total=False) to allow partial payloads.
    The storage layer will provide sensible defaults for missing fields.
    """

    # Core request identification
    request_id: str
    timestamp: int | float | datetime

    # Request details
    method: str
    endpoint: str
    path: str
    query: str
    client_ip: str
    user_agent: str

    # Service and model info
    service_type: str
    model: str
    streaming: bool

    # Response details
    status_code: int
    duration_ms: float
    duration_seconds: float

    # Token and cost tracking
    tokens_input: int
    tokens_output: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    cost_sdk_usd: float


class SimpleDuckDBStorage:
    """Simple DuckDB storage with queue-based writes to prevent deadlocks."""

    def __init__(self, database_path: str | Path = "data/metrics.duckdb"):
        """Initialize simple DuckDB storage.

        Args:
            database_path: Path to DuckDB database file
        """
        self.database_path = Path(database_path)
        self._engine: Engine | None = None
        self._initialized: bool = False
        self._write_queue: asyncio.Queue[AccessLogPayload] = asyncio.Queue()
        self._background_worker_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the storage backend."""
        if self._initialized:
            return

        try:
            # Ensure data directory exists
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

            # Create SQLModel engine
            self._engine = create_engine(f"duckdb:///{self.database_path}")

            # Create schema using SQLModel (synchronous in main thread)
            self._create_schema_sync()

            # Start background worker for queue processing
            self._background_worker_task = asyncio.create_task(
                self._background_worker()
            )

            self._initialized = True
            logger.debug(
                "simple_duckdb_initialized", database_path=str(self.database_path)
            )

        except Exception as e:
            logger.error("simple_duckdb_init_error", error=str(e), exc_info=True)
            raise

    def _create_schema_sync(self) -> None:
        """Create database schema using SQLModel (synchronous)."""
        if not self._engine:
            return

        try:
            # Create tables using SQLModel metadata
            SQLModel.metadata.create_all(self._engine)
            logger.debug("duckdb_schema_created")

        except Exception as e:
            logger.error("simple_duckdb_schema_error", error=str(e))
            raise

    async def _ensure_query_column(self) -> None:
        """Ensure query column exists in the access_logs table."""
        if not self._engine:
            return

        try:
            with Session(self._engine) as session:
                # Check if query column exists
                result = session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = 'access_logs' AND column_name = 'query'"
                    )
                )
                if not result.fetchone():
                    # Add query column if it doesn't exist
                    session.execute(
                        text(
                            "ALTER TABLE access_logs ADD COLUMN query VARCHAR DEFAULT ''"
                        )
                    )
                    session.commit()
                    logger.info("Added query column to access_logs table")

        except Exception as e:
            logger.warning("Failed to check/add query column", error=str(e))
            # Continue without failing - the column might already exist or schema might be different

    async def store_request(self, data: AccessLogPayload) -> bool:
        """Store a single request log entry asynchronously via queue.

        Args:
            data: Request data to store

        Returns:
            True if queued successfully
        """
        if not self._initialized:
            return False

        try:
            # Add to queue for background processing
            await self._write_queue.put(data)
            return True
        except Exception as e:
            logger.error(
                "queue_store_error",
                error=str(e),
                request_id=data.get("request_id"),
            )
            return False

    async def _background_worker(self) -> None:
        """Background worker to process queued write operations sequentially."""
        logger.debug("duckdb_background_worker_started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for either a queue item or shutdown with timeout
                try:
                    data = await asyncio.wait_for(self._write_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue  # Check shutdown event and continue

                # Process the queued write operation synchronously
                try:
                    success = self._store_request_sync(data)
                    if success:
                        logger.debug(
                            "queue_processed_successfully",
                            request_id=data.get("request_id"),
                        )
                except Exception as e:
                    logger.error(
                        "background_worker_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=True,
                    )
                finally:
                    # Always mark the task as done, regardless of success/failure
                    self._write_queue.task_done()

            except Exception as e:
                logger.error(
                    "background_worker_unexpected_error",
                    error=str(e),
                    exc_info=True,
                )
                # Continue processing other items

        # Process any remaining items in the queue during shutdown
        logger.debug("processing_remaining_queue_items_on_shutdown")
        while not self._write_queue.empty():
            try:
                # Get remaining items without timeout during shutdown
                data = self._write_queue.get_nowait()

                # Process the queued write operation synchronously
                try:
                    success = self._store_request_sync(data)
                    if success:
                        logger.debug(
                            "shutdown_queue_processed_successfully",
                            request_id=data.get("request_id"),
                        )
                except Exception as e:
                    logger.error(
                        "shutdown_background_worker_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=True,
                    )
                finally:
                    # Always mark the task as done, regardless of success/failure
                    self._write_queue.task_done()

            except asyncio.QueueEmpty:
                # No more items to process
                break
            except Exception as e:
                logger.error(
                    "shutdown_background_worker_unexpected_error",
                    error=str(e),
                    exc_info=True,
                )
                # Continue processing other items

        logger.debug("duckdb_background_worker_stopped")

    def _store_request_sync(self, data: AccessLogPayload) -> bool:
        """Synchronous version of store_request for thread pool execution."""
        try:
            # Convert Unix timestamp to datetime if needed
            timestamp_value = data.get("timestamp", time.time())
            if isinstance(timestamp_value, int | float):
                timestamp_dt = datetime.fromtimestamp(timestamp_value)
            else:
                timestamp_dt = timestamp_value

            # Create AccessLog object with type validation
            access_log = AccessLog(
                request_id=data.get("request_id", ""),
                timestamp=timestamp_dt,
                method=data.get("method", ""),
                endpoint=data.get("endpoint", ""),
                path=data.get("path", data.get("endpoint", "")),
                query=data.get("query", ""),
                client_ip=data.get("client_ip", ""),
                user_agent=data.get("user_agent", ""),
                service_type=data.get("service_type", ""),
                model=data.get("model", ""),
                streaming=data.get("streaming", False),
                status_code=data.get("status_code", 200),
                duration_ms=data.get("duration_ms", 0.0),
                duration_seconds=data.get("duration_seconds", 0.0),
                tokens_input=data.get("tokens_input", 0),
                tokens_output=data.get("tokens_output", 0),
                cache_read_tokens=data.get("cache_read_tokens", 0),
                cache_write_tokens=data.get("cache_write_tokens", 0),
                cost_usd=data.get("cost_usd", 0.0),
                cost_sdk_usd=data.get("cost_sdk_usd", 0.0),
            )

            # Store using SQLModel session
            with Session(self._engine) as session:
                # Add new log entry (no merge needed as each request is unique)
                session.add(access_log)
                session.commit()

            logger.info(
                "simple_duckdb_store_success",
                request_id=data.get("request_id"),
                service_type=data.get("service_type", ""),
                model=data.get("model", ""),
                tokens_input=data.get("tokens_input", 0),
                tokens_output=data.get("tokens_output", 0),
                cost_usd=data.get("cost_usd", 0.0),
                endpoint=data.get("endpoint", ""),
                timestamp=timestamp_dt.isoformat() if timestamp_dt else None,
            )
            return True

        except Exception as e:
            logger.error(
                "simple_duckdb_store_error",
                error=str(e),
                request_id=data.get("request_id"),
            )
            return False

    async def store_batch(self, metrics: Sequence[AccessLogPayload]) -> bool:
        """Store a batch of metrics efficiently.

        Args:
            metrics: List of metric data to store

        Returns:
            True if batch stored successfully
        """
        if not self._initialized or not metrics or not self._engine:
            return False

        try:
            # Store using SQLModel with upsert behavior
            with Session(self._engine) as session:
                for metric in metrics:
                    # Convert Unix timestamp to datetime if needed
                    timestamp_value = metric.get("timestamp", time.time())
                    if isinstance(timestamp_value, int | float):
                        timestamp_dt = datetime.fromtimestamp(timestamp_value)
                    else:
                        timestamp_dt = timestamp_value

                    # Create AccessLog object with type validation
                    access_log = AccessLog(
                        request_id=metric.get("request_id", ""),
                        timestamp=timestamp_dt,
                        method=metric.get("method", ""),
                        endpoint=metric.get("endpoint", ""),
                        path=metric.get("path", metric.get("endpoint", "")),
                        query=metric.get("query", ""),
                        client_ip=metric.get("client_ip", ""),
                        user_agent=metric.get("user_agent", ""),
                        service_type=metric.get("service_type", ""),
                        model=metric.get("model", ""),
                        streaming=metric.get("streaming", False),
                        status_code=metric.get("status_code", 200),
                        duration_ms=metric.get("duration_ms", 0.0),
                        duration_seconds=metric.get("duration_seconds", 0.0),
                        tokens_input=metric.get("tokens_input", 0),
                        tokens_output=metric.get("tokens_output", 0),
                        cache_read_tokens=metric.get("cache_read_tokens", 0),
                        cache_write_tokens=metric.get("cache_write_tokens", 0),
                        cost_usd=metric.get("cost_usd", 0.0),
                        cost_sdk_usd=metric.get("cost_sdk_usd", 0.0),
                    )
                    # Use merge to handle potential duplicates
                    session.merge(access_log)

                session.commit()

            logger.info(
                "simple_duckdb_batch_store_success",
                batch_size=len(metrics),
                service_types=[
                    m.get("service_type", "") for m in metrics[:3]
                ],  # First 3 for sampling
                request_ids=[
                    m.get("request_id", "") for m in metrics[:3]
                ],  # First 3 for sampling
            )
            return True

        except Exception as e:
            logger.error(
                "simple_duckdb_store_batch_error",
                error=str(e),
                metric_count=len(metrics),
            )
            return False

    async def store(self, metric: AccessLogPayload) -> bool:
        """Store single metric.

        Args:
            metric: Metric data to store

        Returns:
            True if stored successfully
        """
        return await self.store_batch([metric])

    async def query(
        self,
        sql: str,
        params: dict[str, Any] | list[Any] | None = None,
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
        if not self._initialized or not self._engine:
            return []

        try:
            # Use SQLModel for querying
            with Session(self._engine) as session:
                # For now, we'll use raw SQL through the engine
                # In a full implementation, this would be converted to SQLModel queries

                # Use parameterized query to prevent SQL injection
                limited_sql = "SELECT * FROM (" + sql + ") LIMIT :limit"

                query_params = {"limit": limit}
                if params:
                    # Merge user params with limit param
                    if isinstance(params, dict):
                        query_params.update(params)
                        result = session.execute(text(limited_sql), query_params)
                    else:
                        # If params is a list, we need to handle it differently
                        # For now, we'll use the safer approach of not supporting list params with limits
                        result = session.execute(text(sql), params)
                else:
                    result = session.execute(text(limited_sql), query_params)

                # Convert to list of dictionaries
                columns = list(result.keys())
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
        if not self._engine:
            return []

        try:
            with Session(self._engine) as session:
                statement = (
                    select(AccessLog).order_by(desc(AccessLog.timestamp)).limit(limit)
                )
                results = session.exec(statement).all()
                return [log.dict() for log in results]
        except Exception as e:
            logger.error("sqlmodel_query_error", error=str(e))
            return []

    async def get_analytics(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        model: str | None = None,
        service_type: str | None = None,
    ) -> dict[str, Any]:
        """Get analytics using SQLModel.

        Args:
            start_time: Start timestamp (Unix time)
            end_time: End timestamp (Unix time)
            model: Filter by model name
            service_type: Filter by service type

        Returns:
            Analytics summary data
        """
        if not self._engine:
            return {}

        try:
            with Session(self._engine) as session:
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
                    statement = statement.where(AccessLog.service_type == service_type)

                # Get summary statistics using individual queries to avoid overload issues
                base_where_conditions = []
                if start_time:
                    start_dt = datetime.fromtimestamp(start_time)
                    base_where_conditions.append(AccessLog.timestamp >= start_dt)
                if end_time:
                    end_dt = datetime.fromtimestamp(end_time)
                    base_where_conditions.append(AccessLog.timestamp <= end_dt)
                if model:
                    base_where_conditions.append(AccessLog.model == model)
                if service_type:
                    base_where_conditions.append(AccessLog.service_type == service_type)

                total_requests = session.exec(
                    select(func.count())
                    .select_from(AccessLog)
                    .where(*base_where_conditions)
                ).first()

                avg_duration = session.exec(
                    select(func.avg(AccessLog.duration_ms))
                    .select_from(AccessLog)
                    .where(*base_where_conditions)
                ).first()

                total_cost = session.exec(
                    select(func.sum(AccessLog.cost_usd))
                    .select_from(AccessLog)
                    .where(*base_where_conditions)
                ).first()

                total_tokens_input = session.exec(
                    select(func.sum(AccessLog.tokens_input))
                    .select_from(AccessLog)
                    .where(*base_where_conditions)
                ).first()

                total_tokens_output = session.exec(
                    select(func.sum(AccessLog.tokens_output))
                    .select_from(AccessLog)
                    .where(*base_where_conditions)
                ).first()

                return {
                    "summary": {
                        "total_requests": total_requests or 0,
                        "avg_duration_ms": avg_duration or 0,
                        "total_cost_usd": total_cost or 0,
                        "total_tokens_input": total_tokens_input or 0,
                        "total_tokens_output": total_tokens_output or 0,
                    },
                    "query_time": time.time(),
                }

        except Exception as e:
            logger.error("sqlmodel_analytics_error", error=str(e))
            return {}

    async def close(self) -> None:
        """Close the database connection and stop background worker."""
        # Signal shutdown to background worker
        self._shutdown_event.set()

        # Wait for background worker to finish
        if self._background_worker_task:
            try:
                await asyncio.wait_for(self._background_worker_task, timeout=5.0)
            except TimeoutError:
                logger.warning("background_worker_shutdown_timeout")
                self._background_worker_task.cancel()
            except Exception as e:
                logger.error("background_worker_shutdown_error", error=str(e))

        # Process remaining items in queue (with timeout)
        try:
            await asyncio.wait_for(self._write_queue.join(), timeout=2.0)
        except TimeoutError:
            logger.warning(
                "queue_drain_timeout", remaining_items=self._write_queue.qsize()
            )

        if self._engine:
            try:
                self._engine.dispose()
            except Exception as e:
                logger.error("simple_duckdb_engine_close_error", error=str(e))
            finally:
                self._engine = None

        self._initialized = False

    def is_enabled(self) -> bool:
        """Check if storage is enabled and available."""
        return self._initialized

    async def health_check(self) -> dict[str, Any]:
        """Get health status of the storage backend."""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "enabled": False,
            }

        try:
            if self._engine:
                with Session(self._engine) as session:
                    statement = select(func.count()).select_from(AccessLog)
                    access_log_count = session.exec(statement).first()

                    return {
                        "status": "healthy",
                        "enabled": True,
                        "database_path": str(self.database_path),
                        "access_log_count": access_log_count,
                        "backend": "sqlmodel",
                    }
            else:
                return {
                    "status": "no_connection",
                    "enabled": False,
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "enabled": False,
                "error": str(e),
            }

    async def reset_data(self) -> bool:
        """Reset all data in the storage (useful for testing/debugging).

        Returns:
            True if reset was successful
        """
        if not self._initialized or not self._engine:
            return False

        try:
            # Run the reset operation in a thread pool
            return await asyncio.to_thread(self._reset_data_sync)
        except Exception as e:
            logger.error("simple_duckdb_reset_error", error=str(e))
            return False

    def _reset_data_sync(self) -> bool:
        """Synchronous version of reset_data for thread pool execution."""
        try:
            with Session(self._engine) as session:
                # Delete all records from access_logs table
                session.execute(text("DELETE FROM access_logs"))
                session.commit()

            logger.info("simple_duckdb_reset_success")
            return True
        except Exception as e:
            logger.error("simple_duckdb_reset_sync_error", error=str(e))
            return False
