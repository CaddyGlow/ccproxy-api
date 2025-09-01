"""Shared dependencies for CCProxy API Server."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import httpx
from fastapi import Depends, Request

from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.logging import get_logger
from ccproxy.hooks import HookManager
from ccproxy.services.container import ServiceContainer
from plugins.duckdb_storage.storage import SimpleDuckDBStorage


if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


def get_cached_settings(request: Request) -> Settings:
    """Get cached settings from app state.

    This avoids recomputing settings on every request by using the
    settings instance computed during application startup.

    Args:
        request: FastAPI request object

    Returns:
        Settings instance from app state

    Raises:
        RuntimeError: If settings are not available in app state
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        # Fallback to get_settings() for safety, but this should not happen
        # in normal operation after lifespan startup
        logger.warning(
            "Settings not found in app state, falling back to get_settings()",
            category="lifecycle",
        )
        settings = get_settings()
    return settings


async def get_http_client(request: Request) -> httpx.AsyncClient:
    """Get container-managed HTTP client from the service container.

    Falls back to creating a container if missing on app state (logs warning).

    Returns:
        Shared httpx.AsyncClient managed by ServiceContainer
    """
    logger.debug("getting_http_client_from_container", category="lifecycle")
    container: ServiceContainer | None = getattr(request.app.state, "service_container", None)
    if container is None:
        # Fallback: create and attach a container to avoid runtime failures
        settings = getattr(request.app.state, "settings", None) or get_settings()
        logger.warning("service_container_missing_on_app_state_created", category="lifecycle")
        container = ServiceContainer(settings)
        request.app.state.service_container = container
    return container.get_http_client()


# ProxyService removed - use ServiceContainer directly for dependency injection


class _NullObservabilityMetrics:
    """Null metrics stub for backward compatibility.

    Provides `is_enabled()` and `registry` attributes used by routes.
    """

    registry: Any | None = None

    def is_enabled(self) -> bool:  # noqa: D401
        """Return False to indicate metrics are disabled without plugin."""
        return False


def get_observability_metrics() -> Any:
    """Get observability metrics instance.

    Metrics are handled by the metrics plugin. This returns a safe stub
    so core routes that reference metrics can guard behavior without errors.

    Returns:
        None (metrics handled by plugin)
    """
    logger.debug("metrics_stub_returned", category="lifecycle")
    return _NullObservabilityMetrics()


async def get_log_storage(request: Request) -> SimpleDuckDBStorage | None:
    """Get log storage from app state.

    Args:
        request: FastAPI request object

    Returns:
        SimpleDuckDBStorage instance if available, None otherwise
    """
    return getattr(request.app.state, "log_storage", None)


async def get_duckdb_storage(request: Request) -> SimpleDuckDBStorage | None:
    """Get DuckDB storage from app state (backward compatibility).

    Args:
        request: FastAPI request object

    Returns:
        SimpleDuckDBStorage instance if available, None otherwise
    """
    # Try new name first, then fall back to old name for backward compatibility
    storage = getattr(request.app.state, "log_storage", None)
    if storage is None:
        storage = getattr(request.app.state, "duckdb_storage", None)
    return storage


def get_hook_manager(request: Request) -> HookManager | None:
    """Get hook manager from app state.

    Args:
        request: FastAPI request object

    Returns:
        HookManager instance if available, None otherwise
    """
    return getattr(request.app.state, "hook_manager", None)


# V2 Plugin system dependencies
def get_plugin_adapter(plugin_name: str) -> Any:
    """Create a dependency function for a specific plugin's adapter.

    Args:
        plugin_name: Name of the plugin

    Returns:
        Dependency function that retrieves the plugin's adapter
    """
    from fastapi import HTTPException

    from ccproxy.services.adapters.base import BaseAdapter

    def _get_adapter(request: Request) -> BaseAdapter:
        """Get adapter for the specified plugin.

        Args:
            request: FastAPI request object

        Returns:
            Plugin adapter instance

        Raises:
            HTTPException: If plugin or adapter not available
        """
        if not hasattr(request.app.state, "plugin_registry"):
            raise HTTPException(
                status_code=503, detail="Plugin registry not initialized"
            )

        from ccproxy.plugins.factory import PluginRegistry
        from ccproxy.plugins.runtime import ProviderPluginRuntime

        registry: PluginRegistry = request.app.state.plugin_registry
        runtime = registry.get_runtime(plugin_name)

        if not runtime:
            raise HTTPException(
                status_code=503, detail=f"Plugin {plugin_name} not initialized"
            )

        if not isinstance(runtime, ProviderPluginRuntime):
            raise HTTPException(
                status_code=503, detail=f"Plugin {plugin_name} is not a provider plugin"
            )

        if not runtime.adapter:
            raise HTTPException(
                status_code=503, detail=f"Plugin {plugin_name} adapter not available"
            )

        # Cast is safe because we've verified runtime is ProviderPluginRuntime
        adapter: BaseAdapter = runtime.adapter
        return adapter

    return _get_adapter


# Type aliases for service dependencies
SettingsDep = Annotated[Settings, Depends(get_cached_settings)]
# ProxyServiceDep removed - ProxyService no longer used
HTTPClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
ObservabilityMetricsDep = Annotated[Any, Depends(get_observability_metrics)]
LogStorageDep = Annotated[SimpleDuckDBStorage | None, Depends(get_log_storage)]
DuckDBStorageDep = Annotated[SimpleDuckDBStorage | None, Depends(get_duckdb_storage)]
HookManagerDep = Annotated[HookManager | None, Depends(get_hook_manager)]

# Plugin-specific adapter dependencies are declared in each plugin's routes module
