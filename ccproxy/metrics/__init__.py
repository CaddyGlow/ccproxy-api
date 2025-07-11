"""Metrics collection and storage module."""

from .exporters import (
    BaseMetricsExporter,
    ExporterConnectionError,
    ExporterError,
    ExporterTimeoutError,
    ExporterValidationError,
    JsonApiExporter,
    PrometheusExporter,
)
from .models import AggregatedMetrics, MetricType, RequestMetric
from .storage import (
    InMemoryMetricsStorage,
    MetricsStorage,
    StorageConnectionError,
    StorageError,
    StorageInitializationError,
    StorageIntegrityError,
    StorageOperationError,
)


# Optional storage backends
try:
    from .storage import SQLiteMetricsStorage

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    SQLiteMetricsStorage = None  # type: ignore

# PostgreSQL storage is not yet implemented
_POSTGRES_AVAILABLE = False
PostgreSQLMetricStorage = None


__all__ = [
    # Models
    "RequestMetric",
    "AggregatedMetrics",
    "MetricType",
    # Storage base
    "MetricsStorage",
    "StorageError",
    "StorageConnectionError",
    "StorageInitializationError",
    "StorageOperationError",
    "StorageIntegrityError",
    # Storage implementations
    "InMemoryMetricsStorage",
    # Exporters
    "BaseMetricsExporter",
    "ExporterError",
    "ExporterConnectionError",
    "ExporterValidationError",
    "ExporterTimeoutError",
    "PrometheusExporter",
    "JsonApiExporter",
]

# Add optional storage backends if available
if _SQLITE_AVAILABLE and SQLiteMetricsStorage is not None:
    __all__.append("SQLiteMetricsStorage")

# PostgreSQL storage will be added to __all__ when implemented
