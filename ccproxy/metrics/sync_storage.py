"""Synchronous storage layer for metrics persistence (fallback for environments without greenlet)."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import and_, case, create_engine, delete, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

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


class SyncMetricsStorage:
    """Synchronous storage layer for metrics persistence."""

    def __init__(self, database_url: str = "sqlite:///metrics.db"):
        """Initialize metrics storage.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
            if "sqlite" in database_url
            else {},
        )
        self.session_factory = sessionmaker(bind=self.engine)

    def initialize(self) -> None:
        """Initialize the database and create tables if they don't exist."""
        try:
            # Create database directory if using SQLite
            if "sqlite" in self.database_url:
                db_path = self.database_url.replace("sqlite:///", "")
                db_file = Path(db_path)
                db_file.parent.mkdir(parents=True, exist_ok=True)

            # Create tables
            Base.metadata.create_all(self.engine)

            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()

    def store_metrics_snapshot(
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

        with self.session_factory() as session:
            session.add(snapshot)
            session.commit()

    def store_request_log(
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
            cost_dollars=model_metrics.estimated_cost if model_metrics else 0.0,
            user_agent_category=http_metrics.user_agent_category.value,
            error_type=error_metrics.error_type if error_metrics else None,
        )

        with self.session_factory() as session:
            session.add(request_log)
            session.commit()

    def get_metrics_snapshots(
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
        with self.session_factory() as session:
            query = select(MetricsSnapshot)

            if metric_name:
                query = query.where(MetricsSnapshot.metric_name == metric_name)
            if start_time:
                query = query.where(MetricsSnapshot.timestamp >= start_time)
            if end_time:
                query = query.where(MetricsSnapshot.timestamp <= end_time)

            query = query.order_by(desc(MetricsSnapshot.timestamp)).limit(limit)

            result = session.execute(query)
            return list(result.scalars().all())

    def get_request_logs(
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
        with self.session_factory() as session:
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

            result = session.execute(query)
            return list(result.scalars().all())

    def calculate_daily_aggregates(self, date: datetime) -> None:
        """Calculate and store daily aggregates for a given date.

        Args:
            date: Date to calculate aggregates for
        """
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        with self.session_factory() as session:
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

            result = session.execute(query)
            aggregates = result.fetchall()

            # Delete existing aggregates for the date
            session.execute(
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

            session.commit()
            logger.info(
                f"Calculated daily aggregates for {date.date()} - {len(aggregates)} records"
            )

    def get_daily_aggregates(
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
        with self.session_factory() as session:
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

            result = session.execute(query)
            return list(result.scalars().all())

    def cleanup_old_data(self, retention_days: int = 30) -> None:
        """Clean up old data based on retention policy.

        Args:
            retention_days: Number of days to retain data
        """
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        with self.session_factory() as session:
            # Clean up old metrics snapshots
            session.execute(
                delete(MetricsSnapshot).where(MetricsSnapshot.timestamp < cutoff_date)
            )

            # Clean up old request logs
            session.execute(
                delete(RequestLog).where(RequestLog.timestamp < cutoff_date)
            )

            # Clean up old daily aggregates (keep aggregates longer than individual logs)
            agg_cutoff_date = datetime.utcnow() - timedelta(days=retention_days * 3)
            session.execute(
                delete(DailyAggregate).where(DailyAggregate.date < agg_cutoff_date)
            )

            session.commit()
            logger.info(f"Cleaned up data older than {retention_days} days")

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary containing storage statistics
        """
        with self.session_factory() as session:
            # Count records in each table
            metrics_count = session.execute(
                select(func.count(MetricsSnapshot.id))
            ).scalar()

            request_logs_count = session.execute(
                select(func.count(RequestLog.id))
            ).scalar()

            daily_agg_count = session.execute(
                select(func.count(DailyAggregate.id))
            ).scalar()

            # Get date ranges
            oldest_request = session.execute(
                select(func.min(RequestLog.timestamp))
            ).scalar()

            newest_request = session.execute(
                select(func.max(RequestLog.timestamp))
            ).scalar()

            return {
                "metrics_snapshots_count": metrics_count or 0,
                "request_logs_count": request_logs_count or 0,
                "daily_aggregates_count": daily_agg_count or 0,
                "oldest_request": oldest_request,
                "newest_request": newest_request,
                "database_url": self.database_url,
            }


# Global storage instance
_sync_storage_instance: SyncMetricsStorage | None = None


def get_sync_metrics_storage(
    database_url: str = "sqlite:///metrics.db",
) -> SyncMetricsStorage:
    """Get or create the global synchronous metrics storage instance.

    Args:
        database_url: Database connection URL

    Returns:
        SyncMetricsStorage instance
    """
    global _sync_storage_instance
    if _sync_storage_instance is None:
        _sync_storage_instance = SyncMetricsStorage(database_url)
        _sync_storage_instance.initialize()
    return _sync_storage_instance


def close_sync_metrics_storage() -> None:
    """Close the global synchronous metrics storage instance."""
    global _sync_storage_instance
    if _sync_storage_instance:
        _sync_storage_instance.close()
        _sync_storage_instance = None
