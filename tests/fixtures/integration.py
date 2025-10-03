"""Fast integration test fixtures for plugin testing.

Provides reusable, high-performance fixtures for testing CCProxy plugins
with minimal startup overhead and proper isolation.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.config.core import LoggingSettings, ServerSettings
from ccproxy.config.settings import Settings
from ccproxy.services.container import ServiceContainer


PLUGINS_DIR = Path(__file__).resolve().parents[2] / "ccproxy" / "plugins"


def _available_plugin_names() -> set[str]:
    """Return the set of filesystem plugin package names."""

    if not PLUGINS_DIR.exists():
        return set()

    return {
        entry.name
        for entry in PLUGINS_DIR.iterdir()
        if entry.is_dir() and (entry / "plugin.py").exists()
    }


def _build_isolated_plugin_settings(
    plugin_configs: dict[str, dict[str, Any]],
    *,
    logging_overrides: dict[str, Any] | None = None,
    extra_disabled: Iterable[str] | None = None,
) -> Settings:
    """Create test settings that only enable the explicitly requested plugins."""

    requested_plugins = set(plugin_configs.keys())

    # Always disable DuckDB storage unless explicitly requested (avoids I/O).
    if "duckdb_storage" not in plugin_configs:
        plugin_configs = {
            "duckdb_storage": {"enabled": False},
            **plugin_configs,
        }
        requested_plugins.add("duckdb_storage")

    # Ensure request tracer is present with raw logging disabled by default
    request_tracer_cfg = plugin_configs.get("request_tracer", {})
    request_tracer_cfg.setdefault("enabled", True)
    request_tracer_cfg.setdefault("json_logs_enabled", False)
    request_tracer_cfg.setdefault("raw_http_enabled", False)
    plugin_configs["request_tracer"] = request_tracer_cfg
    requested_plugins.add("request_tracer")

    disabled_plugins = sorted(_available_plugin_names() - requested_plugins)

    if extra_disabled:
        disabled_plugins.extend(extra_disabled)

    enabled_plugins = sorted(requested_plugins)

    return Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,
        enabled_plugins=enabled_plugins,
        disabled_plugins=disabled_plugins,
        plugins=plugin_configs,
        logging=LoggingSettings(
            level="ERROR",
            verbose_api=False,
            **(logging_overrides or {}),
        ),
    )


@pytest.fixture(scope="session")
def base_integration_settings() -> Settings:
    """Base settings for integration tests with minimal overhead."""
    return Settings(
        enable_plugins=False,  # Disable all plugins by default
        plugins={},  # Empty plugin configuration
        # Disable expensive features for faster tests
        logging=LoggingSettings(
            level="ERROR",  # Minimal logging for speed
            verbose_api=False,
        ),
        # Minimal server config
        server=ServerSettings(
            host="127.0.0.1",
            port=8000,  # Use standard port for tests
        ),
    )


@pytest.fixture(scope="session")
def base_service_container(
    base_integration_settings: Settings,
) -> ServiceContainer:
    """Shared service container for integration tests."""
    return create_service_container(base_integration_settings)


@pytest.fixture
def integration_app_factory():
    """Factory for creating FastAPI apps with plugin configurations."""

    async def _create_app(plugin_configs: dict[str, dict[str, Any]]) -> FastAPI:
        """Create app with specific plugin configuration.

        Args:
            plugin_configs: Dict mapping plugin names to their configuration
                          e.g., {"metrics": {"enabled": True, "metrics_endpoint_enabled": True}}
        """
        # Set up logging manually for test environment - minimal logging for speed
        from ccproxy.core.logging import setup_logging

        setup_logging(json_logs=False, log_level_name="ERROR")

        # Explicitly disable known default-on system plugins that can cause I/O
        # side effects in isolated test environments unless requested.
        settings = _build_isolated_plugin_settings(plugin_configs)

        service_container = create_service_container(settings)
        app = create_app(service_container)
        await initialize_plugins_startup(app, settings)

        return app

    return _create_app


@pytest.fixture
def integration_client_factory(integration_app_factory):
    """Factory for creating HTTP clients with plugin configurations."""

    async def _create_client(plugin_configs: dict[str, dict[str, Any]]) -> AsyncClient:
        """Create HTTP client with specific plugin configuration."""
        app = await integration_app_factory(plugin_configs)

        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _create_client


@pytest.fixture(scope="session")
def metrics_integration_app():
    """Pre-configured app for metrics plugin integration tests - session scoped."""
    from ccproxy.core.logging import setup_logging

    # Set up logging manually for test environment - minimal logging for speed
    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = _build_isolated_plugin_settings(
        {
            "metrics": {
                "enabled": True,
                "metrics_endpoint_enabled": True,
            }
        }
    )

    service_container = create_service_container(settings)
    # Create the app once per session
    return create_app(service_container), settings


@pytest.fixture
async def metrics_integration_client(metrics_integration_app):
    """HTTP client for metrics integration tests - uses shared app."""
    app, settings = metrics_integration_app

    # Initialize plugins async (once per test, but app is shared)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def metrics_custom_integration_app():
    """Pre-configured app for metrics plugin integration tests with custom config - session scoped."""
    from ccproxy.core.logging import setup_logging

    # Set up logging once per session - minimal logging for speed
    setup_logging(json_logs=False, log_level_name="ERROR")

    settings = _build_isolated_plugin_settings(
        {
            "metrics": {
                "enabled": True,
                "metrics_endpoint_enabled": True,
                "include_labels": True,
            }
        }
    )

    service_container = create_service_container(settings)
    return create_app(service_container), settings


@pytest.fixture
async def metrics_custom_integration_client(metrics_custom_integration_app):
    """HTTP client for metrics integration tests with custom configuration - uses shared app."""
    app, settings = metrics_custom_integration_app

    # Initialize plugins async (once per test, but app is shared)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def disabled_plugins_app(base_integration_settings):
    """Pre-configured app with disabled plugins - session scoped."""
    from ccproxy.core.logging import setup_logging

    # Set up logging manually for test environment - minimal logging for speed
    setup_logging(json_logs=False, log_level_name="ERROR")

    # Use base settings which already have plugins disabled
    settings = base_integration_settings
    service_container = create_service_container(settings)

    # Create the app once per session
    return create_app(service_container), settings


@pytest.fixture
async def disabled_plugins_client(disabled_plugins_app):
    """HTTP client with all plugins disabled - uses shared app."""
    app, settings = disabled_plugins_app

    # Initialize plugins async (once per test, but app is shared)
    await initialize_plugins_startup(app, settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# Mock fixtures for external dependencies
@pytest.fixture
def mock_external_apis():
    """Mock external API calls for isolated integration tests."""
    with (
        patch("httpx.AsyncClient.post") as mock_post,
        patch("httpx.AsyncClient.get") as mock_get,
    ):
        # Configure common mock responses
        mock_post.return_value = AsyncMock(
            status_code=200, json=AsyncMock(return_value={})
        )
        mock_get.return_value = AsyncMock(
            status_code=200, json=AsyncMock(return_value={})
        )

        yield {
            "post": mock_post,
            "get": mock_get,
        }


@pytest.fixture
def plugin_integration_markers():
    """Helper for consistent test marking across plugins."""

    def mark_test(plugin_name: str) -> pytest.MarkDecorator:
        """Apply consistent markers to plugin integration tests."""
        return pytest.mark.parametrize("", [()], ids=[f"{plugin_name}_integration"])

    return mark_test
