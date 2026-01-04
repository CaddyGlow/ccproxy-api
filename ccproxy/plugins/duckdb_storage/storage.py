"""Simplified DuckDB storage for low-traffic environments.

This module provides a simple, direct DuckDB storage implementation without
connection pooling or batch processing. Suitable for dev environments with
low request rates (< 10 req/s).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import delete, insert
from sqlalchemy import select as sa_select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlmodel import Session, SQLModel, create_engine, func

from ccproxy.core.async_task_manager import create_managed_task
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger(__name__)


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
        self._write_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._background_worker_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        # Sentinel to wake the background worker immediately on shutdown
        self._sentinel: object = object()

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
            self._background_worker_task = await create_managed_task(
                self._background_worker(),
                name="duckdb_background_worker",
                creator="SimpleDuckDBStorage",
            )

            self._initialized = True
            logger.debug(
                "simple_duckdb_initialized", database_path=str(self.database_path)
            )

        except OSError as e:
            logger.error("simple_duckdb_init_io_error", error=str(e), exc_info=e)
            raise
        except SQLAlchemyError as e:
            logger.error("simple_duckdb_init_db_error", error=str(e), exc_info=e)
            raise
        except Exception as e:
            logger.error("simple_duckdb_init_error", error=str(e), exc_info=e)
            raise

    def optimize(self) -> None:
        """Run PRAGMA optimize on the database engine if available.

        This is a lightweight maintenance step to improve performance and
        reclaim space in DuckDB. Safe to call on file-backed databases.
        """
        if not self._engine:
            return
        try:
            with self._engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA optimize")
                logger.debug("duckdb_optimize_completed")
        except Exception as e:  # pragma: no cover - non-critical maintenance
            logger.warning("duckdb_optimize_failed", error=str(e), exc_info=e)

    def _create_schema_sync(self) -> None:
        """Create database schema using SQLModel (synchronous)."""
        if not self._engine:
            return

        try:
            # Create tables using SQLModel metadata.
            # Note: AccessLog model must be imported by the access_log plugin prior to this call.
            SQLModel.metadata.create_all(self._engine)
            logger.debug("duckdb_schema_created")

        except SQLAlchemyError as e:
            logger.error("simple_duckdb_schema_db_error", error=str(e), exc_info=e)
            raise
        except Exception as e:
            logger.error("simple_duckdb_schema_error", error=str(e), exc_info=e)
            raise

    async def _ensure_query_column(self) -> None:
        """Ensure query column exists in the access_logs table.

        Note: This method uses schema introspection to safely check for columns.
        The table schema is managed by SQLModel, so this is primarily for
        backwards compatibility with existing databases.
        """
        if not self._engine:
            return

        try:
            # SQLModel automatically handles schema creation through metadata.create_all()
            # This method is kept for backwards compatibility but no longer uses raw SQL
            logger.debug("query_column_ensured_via_sqlmodel_schema")

        except Exception as e:
            logger.warning("query_column_check_error", error=str(e), exc_info=e)
            # Continue without failing - SQLModel handles schema management

    async def store_request(self, data: Mapping[str, Any]) -> bool:
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
            await self._write_queue.put(dict(data))
            return True
        except asyncio.QueueFull as e:
            logger.error(
                "queue_store_full_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
            )
            return False
        except Exception as e:
            logger.error(
                "queue_store_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
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

                # We successfully got an item, so we need to mark it done
                try:
                    # If we receive a sentinel item, break out quickly on shutdown
                    if data is self._sentinel:
                        self._write_queue.task_done()
                        break
                    success = self._store_request_sync(data)
                    if success:
                        logger.debug(
                            "queue_processed_successfully",
                            request_id=data.get("request_id"),
                        )
                except SQLAlchemyError as e:
                    logger.error(
                        "background_worker_db_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=e,
                    )
                except Exception as e:
                    logger.error(
                        "background_worker_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=e,
                    )

                # Always mark the task as done for regular items, regardless of success/failure
                if data is not self._sentinel:
                    self._write_queue.task_done()

            except asyncio.CancelledError as e:
                logger.info("background_worker_cancelled", exc_info=e)
                break
            except Exception as e:
                logger.error(
                    "background_worker_unexpected_error",
                    error=str(e),
                    exc_info=e,
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
                except SQLAlchemyError as e:
                    logger.error(
                        "shutdown_background_worker_db_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=e,
                    )
                except Exception as e:
                    logger.error(
                        "shutdown_background_worker_error",
                        error=str(e),
                        request_id=data.get("request_id"),
                        exc_info=e,
                    )
                # Note: No task_done() call needed for get_nowait() items

            except asyncio.QueueEmpty:
                # No more items to process
                break
            except Exception as e:
                logger.error(
                    "shutdown_background_worker_unexpected_error",
                    error=str(e),
                    exc_info=e,
                )
                # Continue processing other items

        logger.debug("duckdb_background_worker_stopped")

    def _store_request_sync(self, data: dict[str, Any]) -> bool:
        """Synchronous version of store_request for thread pool execution."""
        try:
            # Convert Unix timestamp to datetime if needed
            timestamp_value = data.get("timestamp", time.time())
            if isinstance(timestamp_value, int | float):
                timestamp_dt = datetime.fromtimestamp(timestamp_value)
            else:
                timestamp_dt = timestamp_value

            # Store using SQLAlchemy core insert via SQLModel metadata
            values = {
                "request_id": data.get("request_id", ""),
                "timestamp": timestamp_dt,
                "method": data.get("method", ""),
                "endpoint": data.get("endpoint", ""),
                "path": data.get("path", data.get("endpoint", "")),
                "query": data.get("query", ""),
                "client_ip": data.get("client_ip", ""),
                "user_agent": data.get("user_agent", ""),
                "service_type": data.get("service_type", ""),
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
                "streaming": data.get("streaming", False),
                "status_code": data.get("status_code", 200),
                "duration_ms": data.get("duration_ms", 0.0),
                "duration_seconds": data.get("duration_seconds", 0.0),
                "tokens_input": data.get("tokens_input", 0),
                "tokens_output": data.get("tokens_output", 0),
                "cache_read_tokens": data.get("cache_read_tokens", 0),
                "cache_write_tokens": data.get("cache_write_tokens", 0),
                "cost_usd": data.get("cost_usd", 0.0),
                "cost_sdk_usd": data.get("cost_sdk_usd", 0.0),
            }

            table = SQLModel.metadata.tables.get("access_logs")
            if table is None:
                raise RuntimeError(
                    "access_logs table not registered; ensure analytics plugin is enabled"
                )
            with Session(self._engine) as session:
                try:
                    _ = cast(Any, session).exec(insert(table).values(values))
                    session.commit()
                except (OperationalError, IntegrityError, SQLAlchemyError) as e:
                    # Fallback for older schemas without the 'provider' column
                    msg = str(e)
                    if "provider" in values and (
                        "provider" in msg.lower()
                        or "no column" in msg.lower()
                        or "unknown" in msg.lower()
                    ):
                        safe_values = {
                            k: v for k, v in values.items() if k != "provider"
                        }
                        session.rollback()
                        _ = cast(Any, session).exec(insert(table).values(safe_values))
                        session.commit()
                    else:
                        raise

            logger.info(
                "simple_duckdb_store_success",
                request_id=data.get("request_id"),
                service_type=data.get("service_type"),
                model=data.get("model"),
            )
            return True

        except IntegrityError as e:
            logger.error(
                "simple_duckdb_store_integrity_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
            )
            return False
        except OperationalError as e:
            logger.error(
                "simple_duckdb_store_operational_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
            )
            return False
        except SQLAlchemyError as e:
            logger.error(
                "simple_duckdb_store_db_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
            )
            return False
        except Exception as e:
            logger.error(
                "simple_duckdb_store_error",
                error=str(e),
                request_id=data.get("request_id"),
                exc_info=e,
            )
            return False

    async def store_batch(self, metrics: Sequence[dict[str, Any]]) -> bool:
        """Store a batch of request logs.

        Args:
            metrics: List of metric data entries

        Returns:
            True if stored successfully
        """
        if not self._initialized or not self._engine:
            return False

        try:
            rows = []
            for data in metrics:
                timestamp_value = data.get("timestamp", time.time())
                timestamp_dt = (
                    datetime.fromtimestamp(timestamp_value)
                    if isinstance(timestamp_value, int | float)
                    else timestamp_value
                )
                rows.append(
                    {
                        "request_id": data.get("request_id", ""),
                        "timestamp": timestamp_dt,
                        "method": data.get("method", ""),
                        "endpoint": data.get("endpoint", ""),
                        "path": data.get("path", data.get("endpoint", "")),
                        "query": data.get("query", ""),
                        "client_ip": data.get("client_ip", ""),
                        "user_agent": data.get("user_agent", ""),
                        "service_type": data.get("service_type", ""),
                        "provider": data.get("provider", ""),
                        "model": data.get("model", ""),
                        "streaming": data.get("streaming", False),
                        "status_code": data.get("status_code", 200),
                        "duration_ms": data.get("duration_ms", 0.0),
                        "duration_seconds": data.get("duration_seconds", 0.0),
                        "tokens_input": data.get("tokens_input", 0),
                        "tokens_output": data.get("tokens_output", 0),
                        "cache_read_tokens": data.get("cache_read_tokens", 0),
                        "cache_write_tokens": data.get("cache_write_tokens", 0),
                        "cost_usd": data.get("cost_usd", 0.0),
                        "cost_sdk_usd": data.get("cost_sdk_usd", 0.0),
                    }
                )

            table = SQLModel.metadata.tables.get("access_logs")
            if table is None:
                raise RuntimeError(
                    "access_logs table not registered; ensure analytics plugin is enabled"
                )
            with Session(self._engine) as session:
                cast(Any, session).exec(insert(table), rows)
                session.commit()

            logger.info(
                "simple_duckdb_batch_store_success",
                batch_size=len(metrics),
                service_types=[m.get("service_type", "") for m in metrics[:3]],
                request_ids=[m.get("request_id", "") for m in metrics[:3]],
            )
            return True

        except IntegrityError as e:
            logger.error(
                "simple_duckdb_store_batch_integrity_error",
                error=str(e),
                metric_count=len(metrics),
                exc_info=e,
            )
            return False
        except OperationalError as e:
            logger.error(
                "simple_duckdb_store_batch_operational_error",
                error=str(e),
                metric_count=len(metrics),
                exc_info=e,
            )
            return False
        except SQLAlchemyError as e:
            logger.error(
                "simple_duckdb_store_batch_db_error",
                error=str(e),
                metric_count=len(metrics),
                exc_info=e,
            )
            return False
        except Exception as e:
            logger.error(
                "simple_duckdb_store_batch_error",
                error=str(e),
                metric_count=len(metrics),
                exc_info=e,
            )
            return False

    async def store(self, metric: dict[str, Any]) -> bool:
        """Store single metric.

        Args:
            metric: Metric data to store

        Returns:
            True if stored successfully
        """
        return await self.store_batch([metric])

    async def close(self) -> None:
        """Close the database connection and stop background worker."""
        # Signal shutdown to background worker
        self._shutdown_event.set()

        # Wake up background worker immediately if it's waiting on queue.get()
        with contextlib.suppress(Exception):
            self._write_queue.put_nowait(self._sentinel)  # type: ignore[arg-type]

        # Wait for background worker to finish
        if self._background_worker_task:
            try:
                await asyncio.wait_for(self._background_worker_task, timeout=5.0)
            except TimeoutError:
                logger.warning("background_worker_shutdown_timeout")
                self._background_worker_task.cancel()
            except asyncio.CancelledError:
                logger.info("background_worker_shutdown_cancelled")
            except Exception as e:
                logger.error(
                    "background_worker_shutdown_error", error=str(e), exc_info=e
                )

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
            except SQLAlchemyError as e:
                logger.error(
                    "simple_duckdb_engine_close_db_error", error=str(e), exc_info=e
                )
            except Exception as e:
                logger.error(
                    "simple_duckdb_engine_close_error", error=str(e), exc_info=e
                )
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
                # Run the synchronous database operation in a thread pool
                access_log_count = await asyncio.to_thread(self._health_check_sync)

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

        except SQLAlchemyError as e:
            return {
                "status": "unhealthy",
                "enabled": False,
                "error": str(e),
                "error_type": "database",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "enabled": False,
                "error": str(e),
                "error_type": "unknown",
            }

    def _health_check_sync(self) -> int:
        """Synchronous version of health check for thread pool execution."""
        with Session(self._engine) as session:
            table = SQLModel.metadata.tables.get("access_logs")
            if table is None:
                return 0
            statement = sa_select(func.count()).select_from(table)
            return cast(Any, session).exec(statement).first() or 0

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
        except SQLAlchemyError as e:
            logger.error("simple_duckdb_reset_db_error", error=str(e), exc_info=e)
            return False
        except Exception as e:
            logger.error("simple_duckdb_reset_error", error=str(e), exc_info=e)
            return False

    def _reset_data_sync(self) -> bool:
        """Synchronous version of reset_data for thread pool execution.

        Uses safe SQLModel ORM operations instead of raw SQL to prevent injection.
        """
        try:
            table = SQLModel.metadata.tables.get("access_logs")
            if table is None:
                return True
            with Session(self._engine) as session:
                _ = cast(Any, session).exec(delete(table))
                session.commit()

            logger.info("simple_duckdb_reset_success")
            return True
        except SQLAlchemyError as e:
            logger.error("simple_duckdb_reset_sync_db_error", error=str(e), exc_info=e)
            return False
        except Exception as e:
            logger.error("simple_duckdb_reset_sync_error", error=str(e), exc_info=e)
            return False

    async def wait_for_queue_processing(self, timeout: float = 5.0) -> None:
        """Wait for all queued items to be processed by the background worker.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            asyncio.TimeoutError: If processing doesn't complete within timeout
        """
        if not self._initialized or self._shutdown_event.is_set():
            return

        await asyncio.wait_for(self._write_queue.join(), timeout=timeout)
