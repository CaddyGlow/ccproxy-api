"""Metrics collection and storage module."""

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

try:
    from .storage import PostgreSQLMetricStorage

    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False
    PostgreSQLMetricStorage = None  # type: ignore

# Import exporters
from .exporters import (
    BaseMetricsExporter,
    ExporterConnectionError,
    ExporterError,
    ExporterTimeoutError,
    ExporterValidationError,
    JsonApiExporter,
    PrometheusExporter,
)


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

if _POSTGRES_AVAILABLE and PostgreSQLMetricStorage is not None:
    __all__.append("PostgreSQLMetricStorage")
