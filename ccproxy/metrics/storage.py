"""Storage layer for metrics persistence in Claude Code Proxy API Server."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_, case, delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from ccproxy.metrics.database import (
    Base,
    DailyAggregate,
    MetricsSnapshot,
    RequestLog,
    deserialize_labels,
    serialize_labels,
)
from ccproxy.metrics.models import ErrorMetrics, HTTPMetrics, ModelMetrics


logger = logging.getLogger(__name__)


class MetricsStorage:
    """Async storage layer for metrics persistence."""

    def __init__(self, database_url: str = "sqlite+aiosqlite:///metrics.db"):
        """Initialize metrics storage.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
            if "sqlite" in database_url
            else {},
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        try:
            # Create database directory if using SQLite
            if "sqlite" in self.database_url:
                db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
                db_file = Path(db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close the database connection."""
        await self.engine.dispose()

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session.

        Yields:
            AsyncSession: Database session
        """
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def store_metrics_snapshot(
        self,
        metric_name: str,
        metric_type: str,
        labels: dict[str, Any],
        value: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Store a metrics snapshot.

        Args:
            metric_name: Name of the metric
            metric_type: Type of metric (gauge, counter, histogram)
            labels: Metric labels
            value: Metric value
            timestamp: Timestamp of the metric (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        snapshot = MetricsSnapshot(
            timestamp=timestamp,
            metric_name=metric_name,
            metric_type=metric_type,
            labels=serialize_labels(labels),
            value=value,
        )

        async with self.get_session() as session:
            session.add(snapshot)
            await session.commit()

    async def store_request_log(
        self,
        http_metrics: HTTPMetrics,
        model_metrics: ModelMetrics | None = None,
        error_metrics: ErrorMetrics | None = None,
    ) -> None:
        """Store a request log entry.

        Args:
            http_metrics: HTTP request metrics
            model_metrics: Model usage metrics (optional)
            error_metrics: Error metrics (optional)
        """
        request_log = RequestLog(
            timestamp=datetime.utcnow(),
            method=http_metrics.method,
            endpoint=http_metrics.endpoint,
            api_type=http_metrics.api_type,
            model=model_metrics.model if model_metrics else None,
            status_code=http_metrics.status_code,
            duration_ms=http_metrics.duration_seconds * 1000,
            request_size=http_metrics.request_size_bytes,
            response_size=http_metrics.response_size_bytes,
            input_tokens=model_metrics.input_tokens if model_metrics else 0,
            output_tokens=model_metrics.output_tokens if model_metrics else 0,
            cache_read_input_tokens=model_metrics.cache_read_input_tokens
            if model_metrics
            else 0,
            cache_creation_input_tokens=model_metrics.cache_creation_input_tokens
            if model_metrics
            else 0,
            cost_dollars=model_metrics.estimated_cost if model_metrics else 0.0,
            user_agent=http_metrics.user_agent,
            user_agent_category=http_metrics.user_agent_category.value,
            error_type=error_metrics.error_type if error_metrics else None,
            # Rate limit fields
            rate_limit_requests_limit=http_metrics.rate_limit_requests_limit,
            rate_limit_requests_remaining=http_metrics.rate_limit_requests_remaining,
            rate_limit_tokens_limit=http_metrics.rate_limit_tokens_limit,
            rate_limit_tokens_remaining=http_metrics.rate_limit_tokens_remaining,
            rate_limit_reset_timestamp=self._parse_datetime(
                http_metrics.rate_limit_reset_timestamp
            ),
            retry_after_seconds=http_metrics.retry_after_seconds,
            oauth_unified_status=http_metrics.oauth_unified_status,
            oauth_unified_claim=http_metrics.oauth_unified_claim,
            oauth_unified_fallback_percentage=http_metrics.oauth_unified_fallback_percentage,
            oauth_unified_reset=self._parse_datetime(http_metrics.oauth_unified_reset),
            auth_type=http_metrics.auth_type,
        )

        async with self.get_session() as session:
            session.add(request_log)
            await session.commit()

    def _parse_datetime(self, datetime_str: str | None) -> datetime | None:
        """Parse datetime string to datetime object.

        Args:
            datetime_str: ISO format datetime string

        Returns:
            datetime object or None if parsing fails
        """
        if not datetime_str:
            return None

        try:
            # Parse ISO format datetime string
            return datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse datetime string '{datetime_str}': {e}")
            return None

    async def get_metrics_snapshots(
        self,
        metric_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[MetricsSnapshot]:
        """Get metrics snapshots.

        Args:
            metric_name: Filter by metric name
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of results

        Returns:
            List of metrics snapshots
        """
        async with self.get_session() as session:
            query = select(MetricsSnapshot)

            if metric_name:
                query = query.where(MetricsSnapshot.metric_name == metric_name)
            if start_time:
                query = query.where(MetricsSnapshot.timestamp >= start_time)
            if end_time:
                query = query.where(MetricsSnapshot.timestamp <= end_time)

            query = query.order_by(desc(MetricsSnapshot.timestamp)).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_request_logs(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        endpoint: str | None = None,
        api_type: str | None = None,
        status_code: int | None = None,
        limit: int = 1000,
    ) -> list[RequestLog]:
        """Get request logs.

        Args:
            start_time: Start time filter
            end_time: End time filter
            endpoint: Filter by endpoint
            api_type: Filter by API type
            status_code: Filter by status code
            limit: Maximum number of results

        Returns:
            List of request logs
        """
        async with self.get_session() as session:
            query = select(RequestLog)

            if start_time:
                query = query.where(RequestLog.timestamp >= start_time)
            if end_time:
                query = query.where(RequestLog.timestamp <= end_time)
            if endpoint:
                query = query.where(RequestLog.endpoint == endpoint)
            if api_type:
                query = query.where(RequestLog.api_type == api_type)
            if status_code:
                query = query.where(RequestLog.status_code == status_code)

            query = query.order_by(desc(RequestLog.timestamp)).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def calculate_daily_aggregates(self, date: datetime) -> None:
        """Calculate and store daily aggregates for a given date.

        Args:
            date: Date to calculate aggregates for
        """
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        async with self.get_session() as session:
            # Get aggregated data for the date
            query = (
                select(
                    RequestLog.endpoint,
                    RequestLog.api_type,
                    RequestLog.model,
                    func.count(RequestLog.id).label("total_requests"),
                    func.sum(case((RequestLog.status_code >= 400, 1), else_=0)).label(
                        "total_errors"
                    ),
                    func.avg(RequestLog.duration_ms).label("avg_duration_ms"),
                    func.max(RequestLog.duration_ms).label(
                        "p95_duration_ms"
                    ),  # Simplified for SQLite compatibility
                    func.max(RequestLog.duration_ms).label(
                        "p99_duration_ms"
                    ),  # Simplified for SQLite compatibility
                    func.sum(RequestLog.input_tokens).label("total_input_tokens"),
                    func.sum(RequestLog.output_tokens).label("total_output_tokens"),
                    func.sum(RequestLog.cost_dollars).label("total_cost_dollars"),
                )
                .where(
                    and_(
                        RequestLog.timestamp >= start_date,
                        RequestLog.timestamp < end_date,
                    )
                )
                .group_by(
                    RequestLog.endpoint,
                    RequestLog.api_type,
                    RequestLog.model,
                )
            )

            result = await session.execute(query)
            aggregates = result.fetchall()

            # Delete existing aggregates for the date
            await session.execute(
                delete(DailyAggregate).where(DailyAggregate.date == start_date)
            )

            # Insert new aggregates
            for agg in aggregates:
                daily_agg = DailyAggregate(
                    date=start_date,
                    endpoint=agg.endpoint,
                    api_type=agg.api_type,
                    model=agg.model,
                    total_requests=agg.total_requests,
                    total_errors=agg.total_errors,
                    avg_duration_ms=agg.avg_duration_ms or 0.0,
                    p95_duration_ms=agg.p95_duration_ms or 0.0,
                    p99_duration_ms=agg.p99_duration_ms or 0.0,
                    total_input_tokens=agg.total_input_tokens or 0,
                    total_output_tokens=agg.total_output_tokens or 0,
                    total_cost_dollars=agg.total_cost_dollars or 0.0,
                )
                session.add(daily_agg)

            await session.commit()
            logger.info(
                f"Calculated daily aggregates for {date.date()} - {len(aggregates)} records"
            )

    async def get_daily_aggregates(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        endpoint: str | None = None,
        api_type: str | None = None,
        model: str | None = None,
        limit: int = 1000,
    ) -> list[DailyAggregate]:
        """Get daily aggregates.

        Args:
            start_date: Start date filter
            end_date: End date filter
            endpoint: Filter by endpoint
            api_type: Filter by API type
            model: Filter by model
            limit: Maximum number of results

        Returns:
            List of daily aggregates
        """
        async with self.get_session() as session:
            query = select(DailyAggregate)

            if start_date:
                query = query.where(DailyAggregate.date >= start_date)
            if end_date:
                query = query.where(DailyAggregate.date <= end_date)
            if endpoint:
                query = query.where(DailyAggregate.endpoint == endpoint)
            if api_type:
                query = query.where(DailyAggregate.api_type == api_type)
            if model:
                query = query.where(DailyAggregate.model == model)

            query = query.order_by(desc(DailyAggregate.date)).limit(limit)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def cleanup_old_data(self, retention_days: int = 30) -> None:
        """Clean up old data based on retention policy.

        Args:
            retention_days: Number of days to retain data
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        async with self.get_session() as session:
            # Clean up old metrics snapshots
            await session.execute(
                delete(MetricsSnapshot).where(MetricsSnapshot.timestamp < cutoff_date)
            )

            # Clean up old request logs
            await session.execute(
                delete(RequestLog).where(RequestLog.timestamp < cutoff_date)
            )

            # Clean up old daily aggregates (keep aggregates longer than individual logs)
            agg_cutoff_date = datetime.utcnow() - timedelta(days=retention_days * 3)
            await session.execute(
                delete(DailyAggregate).where(DailyAggregate.date < agg_cutoff_date)
            )

            await session.commit()
            logger.info(f"Cleaned up data older than {retention_days} days")

    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary containing storage statistics
        """
        async with self.get_session() as session:
            # Count records in each table
            metrics_count = await session.execute(
                select(func.count(MetricsSnapshot.id))
            )
            request_logs_count = await session.execute(
                select(func.count(RequestLog.id))
            )
            daily_agg_count = await session.execute(
                select(func.count(DailyAggregate.id))
            )

            # Get date ranges
            oldest_request = await session.execute(
                select(func.min(RequestLog.timestamp))
            )
            newest_request = await session.execute(
                select(func.max(RequestLog.timestamp))
            )

            return {
                "metrics_snapshots_count": metrics_count.scalar() or 0,
                "request_logs_count": request_logs_count.scalar() or 0,
                "daily_aggregates_count": daily_agg_count.scalar() or 0,
                "oldest_request": oldest_request.scalar(),
                "newest_request": newest_request.scalar(),
                "database_url": self.database_url,
            }


# Global storage instance
_storage_instance: MetricsStorage | None = None


async def get_metrics_storage(
    database_url: str = "sqlite+aiosqlite:///metrics.db",
) -> MetricsStorage:
    """Get or create the global metrics storage instance.

    Args:
        database_url: Database connection URL

    Returns:
        MetricsStorage instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = MetricsStorage(database_url)
        await _storage_instance.initialize()
    return _storage_instance


async def close_metrics_storage() -> None:
    """Close the global metrics storage instance."""
    global _storage_instance
    if _storage_instance:
        await _storage_instance.close()
        _storage_instance = None
