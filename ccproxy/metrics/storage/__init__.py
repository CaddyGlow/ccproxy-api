"""Metrics storage backends."""

from .base import (
    MetricsStorage,
    StorageError,
    StorageConnectionError,
    StorageInitializationError,
    StorageOperationError,
    StorageIntegrityError,
)
from .memory import InMemoryMetricsStorage

# Optional SQLite support
try:
    from .sqlite import SQLiteMetricStorage
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    SQLiteMetricStorage = None  # type: ignore

# Optional PostgreSQL support
try:
    from .postgres import PostgreSQLMetricsStorage as PostgreSQLMetricStorage
    _POSTGRES_AVAILABLE = True
except (ImportError, NameError):
    _POSTGRES_AVAILABLE = False
    PostgreSQLMetricStorage = None  # type: ignore


__all__ = [
    # Base classes and exceptions
    "MetricsStorage",
    "StorageError",
    "StorageConnectionError",
    "StorageInitializationError",
    "StorageOperationError",
    "StorageIntegrityError",
    # Storage implementations
    "InMemoryMetricsStorage",
]

# Add SQLite storage if available
if _SQLITE_AVAILABLE and SQLiteMetricStorage:
    __all__.append("SQLiteMetricStorage")

# Add PostgreSQL storage if available
if _POSTGRES_AVAILABLE and PostgreSQLMetricStorage:
    __all__.append("PostgreSQLMetricStorage")