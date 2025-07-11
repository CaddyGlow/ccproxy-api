"""Metrics storage backends."""

from .base import (
    MetricsStorage,
    StorageConnectionError,
    StorageError,
    StorageInitializationError,
    StorageIntegrityError,
    StorageOperationError,
)
from .memory import InMemoryMetricsStorage


# Optional SQLite support
try:
    from .sqlite import SQLiteMetricsStorage

    _SQLITE_AVAILABLE = True
except ImportError:
    _SQLITE_AVAILABLE = False
    SQLiteMetricsStorage = None  # type: ignore


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
