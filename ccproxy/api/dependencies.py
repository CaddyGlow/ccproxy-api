"""Shared dependencies for CCProxy API Server."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, TypeVar

import httpx
from fastapi import Depends, HTTPException, Request

from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_logger
from ccproxy.core.plugins import PluginRegistry, ProviderPluginRuntime
from ccproxy.core.plugins.hooks import HookManager
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.container import ServiceContainer


if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

T = TypeVar("T")


def get_service(service_type: type[T]) -> Callable[[Request], T]:
    """Return a dependency callable that fetches a service from the container."""

    def _get_service(request: Request) -> T:
        """Get a service from the container."""
        container: ServiceContainer | None = getattr(
            request.app.state, "service_container", None
        )
        if container is None:
            logger.error(
                "service_container_missing_on_app_state",
                category="lifecycle",
            )
            raise HTTPException(
                status_code=503, detail="Service container not initialized"
            )
        return container.get_service(service_type)

    return _get_service


def _resolve_service_container(request: Request) -> ServiceContainer | None:
    """Resolve a service container from the request or global context."""

    container: ServiceContainer | None = getattr(
        request.app.state, "service_container", None
    )
    if container is not None:
        return container

    try:
        return ServiceContainer.get_current(strict=False)
    except RuntimeError:
        # Should not happen with strict=False but guard defensively
        return None


def get_cached_settings(request: Request) -> Settings:
    """Get cached settings from app state.

    Raises a 503 HTTPException if no service container is available,
    preserving the existing behaviour for required dependencies.
    """
    return get_service(Settings)(request)


def get_optional_settings(request: Request) -> Settings | None:
    """Best-effort retrieval of settings for optional dependencies.

    Returns a Settings instance if a service container is available.
    Falls back to a new Settings object (with defaults) when running in
    lightweight test contexts where the container is not initialised.
    """

    container = _resolve_service_container(request)
    if container is not None:
        try:
            return container.get_service(Settings)
        except ValueError:
            logger.debug(
                "settings_not_registered_in_container",
                category="config",
            )

    try:
        return Settings()
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning(
            "optional_settings_initialization_failed",
            error=str(exc),
            category="config",
        )
        return None


async def get_http_client(request: Request) -> httpx.AsyncClient:
    """Get container-managed HTTP client from the service container."""
    return get_service(httpx.AsyncClient)(request)


def get_hook_manager(request: Request) -> HookManager:
    """Get HookManager from the service container.

    This dependency is required; if the hook system has not been initialized
    the request will fail with 503 to reflect misconfigured startup order.
    """
    return get_service(HookManager)(request)


def get_plugin_adapter(plugin_name: str) -> Any:
    """Create a dependency function for a specific plugin's adapter."""

    def _get_adapter(request: Request) -> BaseAdapter:
        """Get adapter for the specified plugin."""
        if not hasattr(request.app.state, "plugin_registry"):
            raise HTTPException(
                status_code=503, detail="Plugin registry not initialized"
            )

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

        adapter: BaseAdapter = runtime.adapter
        return adapter

    return _get_adapter


def get_provider_config_dependency(
    plugin_name: str, config_type: type[T]
) -> Callable[[Request], T]:
    """Return a dependency that fetches a provider plugin's validated config."""

    def _get_config(request: Request) -> T:
        if not hasattr(request.app.state, "plugin_registry"):
            raise HTTPException(
                status_code=503, detail="Plugin registry not initialized"
            )

        registry: PluginRegistry = request.app.state.plugin_registry
        runtime = registry.get_runtime(plugin_name)

        if not runtime or not isinstance(runtime, ProviderPluginRuntime):
            raise HTTPException(
                status_code=503, detail=f"Plugin {plugin_name} not initialized"
            )

        context = getattr(runtime, "context", None)
        if not context:
            raise HTTPException(
                status_code=503,
                detail=f"Plugin {plugin_name} configuration unavailable",
            )

        try:
            config = context.get(config_type)
            return config  # type: ignore[no-any-return]
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise HTTPException(
                status_code=503,
                detail=f"Plugin {plugin_name} configuration not loaded",
            ) from exc

    return _get_config


SettingsDep = Annotated[Settings, Depends(get_cached_settings)]
OptionalSettingsDep = Annotated[Settings | None, Depends(get_optional_settings)]
HTTPClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
HookManagerDep = Annotated[HookManager, Depends(get_hook_manager)]
