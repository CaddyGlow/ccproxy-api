"""Metrics collection and storage module."""

from .models import AggregatedMetrics, MetricType, RequestMetric
from .storage import (
    InMemoryMetricsStorage,
    MetricsStorage,
    StorageConnectionError,
    StorageError,
    StorageInitializationError,
    StorageOperationError,
    StorageIntegrityError,
)

# Optional storage backends
try:
    from .storage import SQLiteMetricStorage
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    SQLiteMetricStorage = None  # type: ignore

try:
    from .storage import PostgreSQLMetricStorage
    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False
    PostgreSQLMetricStorage = None  # type: ignore

# Import exporters
from .exporters import (
    BaseMetricsExporter,
    ExporterError,
    ExporterConnectionError,
    ExporterValidationError,
    ExporterTimeoutError,
    PrometheusExporter,
    JsonApiExporter,
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
if _SQLITE_AVAILABLE and SQLiteMetricStorage:
    __all__.append("SQLiteMetricStorage")

if _POSTGRES_AVAILABLE and PostgreSQLMetricStorage:
    __all__.append("PostgreSQLMetricStorage")