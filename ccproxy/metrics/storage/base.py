"""
Abstract base class for metrics storage backends.

This module defines the interface that all metrics storage implementations
must follow.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from ..models import MetricRecord, MetricsSummary, MetricType


class MetricsStorage(ABC):
    """
    Abstract base class for metrics storage backends.

    All storage implementations must inherit from this class and implement
    the required methods.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        This method should set up any necessary connections, create tables,
        or perform other initialization tasks.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage backend and clean up resources.

        This method should close database connections, release locks,
        or perform other cleanup tasks.
        """
        pass

    @abstractmethod
    async def store_metric(self, metric: MetricRecord) -> bool:
        """
        Store a single metric record.

        Args:
            metric: The metric record to store

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def store_metrics(self, metrics: list[MetricRecord]) -> int:
        """
        Store multiple metric records in a batch.

        Args:
            metrics: List of metric records to store

        Returns:
            Number of metrics successfully stored
        """
        pass

    @abstractmethod
    async def get_metric(self, metric_id: UUID) -> MetricRecord | None:
        """
        Retrieve a single metric record by ID.

        Args:
            metric_id: The ID of the metric to retrieve

        Returns:
            The metric record if found, None otherwise
        """
        pass

    @abstractmethod
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
        """
        Retrieve multiple metric records with filtering and pagination.

        Args:
            start_time: Filter metrics after this time
            end_time: Filter metrics before this time
            metric_type: Filter by metric type
            user_id: Filter by user ID
            session_id: Filter by session ID
            request_id: Filter by request ID
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Field to order by (default: timestamp)
            order_desc: Whether to order in descending order
            filters: Additional filters as key-value pairs

        Returns:
            List of metric records matching the criteria
        """
        pass

    @abstractmethod
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
        """
        Count metric records matching the given criteria.

        Args:
            start_time: Filter metrics after this time
            end_time: Filter metrics before this time
            metric_type: Filter by metric type
            user_id: Filter by user ID
            session_id: Filter by session ID
            request_id: Filter by request ID
            filters: Additional filters as key-value pairs

        Returns:
            Number of matching metric records
        """
        pass

    @abstractmethod
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
        """
        Delete metric records matching the given criteria.

        Args:
            start_time: Delete metrics after this time
            end_time: Delete metrics before this time
            metric_type: Filter by metric type
            user_id: Filter by user ID
            session_id: Filter by session ID
            request_id: Filter by request ID
            filters: Additional filters as key-value pairs

        Returns:
            Number of deleted metric records
        """
        pass

    @abstractmethod
    async def get_metrics_summary(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: str | None = None,
        session_id: str | None = None,
        group_by: str | None = None,
    ) -> MetricsSummary:
        """
        Get aggregated metrics summary for a time period.

        Args:
            start_time: Start of time period
            end_time: End of time period
            user_id: Filter by user ID
            session_id: Filter by session ID
            group_by: Group results by field (e.g., 'user_id', 'model')

        Returns:
            Aggregated metrics summary
        """
        pass

    @abstractmethod
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
        """
        Get time series data for metrics.

        Args:
            start_time: Start of time period
            end_time: End of time period
            interval: Time interval for grouping (e.g., '1h', '1d')
            metric_type: Filter by metric type
            aggregation: Aggregation function ('count', 'sum', 'avg', 'min', 'max')
            user_id: Filter by user ID
            session_id: Filter by session ID

        Returns:
            List of time series data points
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the storage backend.

        Returns:
            Dictionary with health status information
        """
        pass

    @abstractmethod
    async def get_storage_info(self) -> dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dictionary with storage backend information
        """
        pass

    # Optional methods that can be overridden for better performance

    async def bulk_insert(self, metrics: list[MetricRecord]) -> int:
        """
        Bulk insert metrics for better performance.

        Default implementation calls store_metrics.
        Subclasses can override for optimized bulk operations.

        Args:
            metrics: List of metric records to insert

        Returns:
            Number of metrics successfully inserted
        """
        return await self.store_metrics(metrics)

    async def vacuum(self) -> None:
        """
        Perform maintenance operations on the storage backend.

        This might include vacuuming, reindexing, or other optimization tasks.
        Default implementation does nothing.
        """
        pass

    async def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the metrics data.

        Args:
            backup_path: Path where to store the backup

        Returns:
            True if backup was successful, False otherwise
        """
        return False

    async def restore(self, backup_path: str) -> bool:
        """
        Restore metrics data from a backup.

        Args:
            backup_path: Path to the backup file

        Returns:
            True if restore was successful, False otherwise
        """
        return False

    async def migrate(self, target_version: str) -> bool:
        """
        Migrate the storage schema to a target version.

        Args:
            target_version: Target schema version

        Returns:
            True if migration was successful, False otherwise
        """
        return False

    async def get_schema_version(self) -> str:
        """
        Get the current schema version.

        Returns:
            Current schema version string
        """
        return "1.0.0"


class StorageError(Exception):
    """Base exception for storage-related errors."""

    pass


class StorageInitializationError(StorageError):
    """Raised when storage initialization fails."""

    pass


class StorageConnectionError(StorageError):
    """Raised when storage connection fails."""

    pass


class StorageOperationError(StorageError):
    """Raised when a storage operation fails."""

    pass


class StorageIntegrityError(StorageError):
    """Raised when data integrity is compromised."""

    pass
