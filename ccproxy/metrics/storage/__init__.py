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
    from .sqlite import SQLiteMetricsStorage
    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    SQLiteMetricsStorage = None  # type: ignore

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
if _SQLITE_AVAILABLE and SQLiteMetricsStorage is not None:
    __all__.append("SQLiteMetricsStorage")

# Add PostgreSQL storage if available
if _POSTGRES_AVAILABLE and PostgreSQLMetricStorage is not None:
    __all__.append("PostgreSQLMetricStorage")