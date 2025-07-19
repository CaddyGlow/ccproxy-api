"""Test factories for creating test fixtures with flexible configurations.

This module provides factory patterns to eliminate combinatorial explosion
in test fixtures by allowing composition of different configurations.
"""

from .fastapi_factory import (
    AppFactoryConfig,
    FastAPIAppFactory,
    FastAPIClientFactory,
    create_auth_app,
    create_mock_claude_app,
    create_unavailable_claude_app,
)


__all__ = [
    "AppFactoryConfig",
    "FastAPIAppFactory",
    "FastAPIClientFactory",
    "create_auth_app",
    "create_mock_claude_app",
    "create_unavailable_claude_app",
]
