"""Test factories for creating test fixtures with flexible configurations.

This module provides factory patterns to eliminate combinatorial explosion
in test fixtures by allowing composition of different configurations.
"""

from .fastapi_factory import (
    FastAPIAppFactory,
    FastAPIClientFactory,
    create_test_app,
)


__all__ = [
    "FastAPIAppFactory",
    "FastAPIClientFactory",
    "create_test_app",
]
