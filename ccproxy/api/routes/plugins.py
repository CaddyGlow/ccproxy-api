"""Plugin management API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette import status

import ccproxy.core.logging
from ccproxy.auth.conditional import ConditionalAuthDep


logger = ccproxy.core.logging.get_logger(__name__)


router = APIRouter(prefix="/plugins", tags=["plugins"])


class PluginInfo(BaseModel):
    """Plugin information model."""

    name: str
    type: str  # "builtin" or "plugin"
    status: str  # "active", "inactive", "error"
    version: str | None = None


class PluginListResponse(BaseModel):
    """Response model for plugin list."""

    plugins: list[PluginInfo]
    total: int


class PluginStatusEntry(BaseModel):
    name: str
    version: str | None = None
    type: str  # "provider" or "system"
    provides: list[str] = []
    requires: list[str] = []
    optional_requires: list[str] = []
    initialized: bool


class PluginStatusResponse(BaseModel):
    initialization_order: list[str]
    services: dict[str, str]  # service_name -> provider plugin
    plugins: list[PluginStatusEntry]


class PluginHealthResponse(BaseModel):
    """Response model for plugin health check."""

    plugin: str
    status: str  # "healthy", "unhealthy", "unknown"
    adapter_loaded: bool
    details: dict[str, Any] | None = None


# Only core plugin management endpoints are exposed:
# - GET /plugins: list loaded plugins
# - GET /plugins/{plugin_name}/health: check plugin health if provided by runtime
# - GET /plugins/status: summarize manifests and initialization state
#
# Dynamic reload/discover/unregister are not supported in v2 and have been removed.


# Plugin registry is accessed directly from app state


@router.get("", response_model=PluginListResponse)
async def list_plugins(
    request: Request,
    auth: ConditionalAuthDep = None,
) -> PluginListResponse:
    """List all loaded plugins and built-in providers.

    Returns:
        List of all available plugins and providers
    """
    plugins: list[PluginInfo] = []

    # Access v2 plugin registry from app state
    if hasattr(request.app.state, "plugin_registry"):
        from ccproxy.core.plugins import PluginRegistry

        registry: PluginRegistry = request.app.state.plugin_registry

        for name in registry.list_plugins():
            factory = registry.get_factory(name)
            if factory:
                from ccproxy.core.plugins import factory_type_name

                manifest = factory.get_manifest()
                plugin_type = factory_type_name(factory)

                plugins.append(
                    PluginInfo(
                        name=name,
                        type=plugin_type,
                        status="active",
                        version=manifest.version,
                    )
                )

    return PluginListResponse(plugins=plugins, total=len(plugins))


@router.get("/{plugin_name}/health", response_model=PluginHealthResponse)
async def plugin_health(
    plugin_name: str,
    request: Request,
    auth: ConditionalAuthDep = None,
) -> PluginHealthResponse:
    """Check the health status of a specific plugin.

    Args:
        plugin_name: Name of the plugin to check

    Returns:
        Health status of the plugin

    Raises:
        HTTPException: If plugin not found
    """
    # Access v2 plugin registry from app state
    if not hasattr(request.app.state, "plugin_registry"):
        raise HTTPException(status_code=503, detail="Plugin registry not initialized")

    from ccproxy.core.plugins import PluginRegistry

    registry: PluginRegistry = request.app.state.plugin_registry

    # Check if plugin exists
    if plugin_name not in registry.list_plugins():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{plugin_name}' not found",
        )

    # Get the plugin runtime instance
    runtime = registry.get_runtime(plugin_name)
    if runtime and hasattr(runtime, "health_check"):
        try:
            health_result = await runtime.health_check()
            # Convert HealthCheckResult to PluginHealthResponse
            # Handle both dict and object response
            if isinstance(health_result, dict):
                status_value = health_result.get("status", "unknown")
                output_value = health_result.get("output")
                version_value = health_result.get("version")
                details_value = health_result.get("details")
            else:
                # Access attributes for non-dict responses
                status_value = getattr(health_result, "status", "unknown")
                output_value = getattr(health_result, "output", None)
                version_value = getattr(health_result, "version", None)
                details_value = getattr(health_result, "details", None)

            return PluginHealthResponse(
                plugin=plugin_name,
                status="healthy"
                if status_value == "pass"
                else "unhealthy"
                if status_value == "fail"
                else "unknown",
                adapter_loaded=True,
                details={
                    "type": "plugin",
                    "active": True,
                    "health_check": {
                        "status": status_value,
                        "output": output_value,
                        "version": version_value,
                        "details": details_value,
                    },
                },
            )
        except (OSError, PermissionError) as e:
            logger.error(
                "plugin_health_check_io_failed",
                plugin=plugin_name,
                error=str(e),
                exc_info=e,
            )
            return PluginHealthResponse(
                plugin=plugin_name,
                status="unhealthy",
                adapter_loaded=True,
                details={"type": "plugin", "active": True, "io_error": str(e)},
            )
        except Exception as e:
            logger.error(
                "plugin_health_check_failed",
                plugin=plugin_name,
                error=str(e),
                exc_info=e,
            )
            return PluginHealthResponse(
                plugin=plugin_name,
                status="unhealthy",
                adapter_loaded=True,
                details={"type": "plugin", "active": True, "error": str(e)},
            )
    else:
        # Plugin doesn't have health check, use basic status
        return PluginHealthResponse(
            plugin=plugin_name,
            status="healthy",
            adapter_loaded=True,
            details={"type": "plugin", "active": True},
        )

        # Endpoints are loaded at startup only


@router.get("/status", response_model=PluginStatusResponse)
async def plugins_status(
    request: Request, auth: ConditionalAuthDep = None
) -> PluginStatusResponse:
    """Get plugin system status, including manifests and init order.

    Returns:
        Initialization order, registered services, and per-plugin manifest summary
    """
    if not hasattr(request.app.state, "plugin_registry"):
        raise HTTPException(status_code=503, detail="Plugin registry not initialized")

    from ccproxy.core.plugins import PluginRegistry

    registry: PluginRegistry = request.app.state.plugin_registry

    # Get manifests and runtime status
    entries: list[PluginStatusEntry] = []
    for name in registry.list_plugins():
        factory = registry.get_factory(name)
        if not factory:
            continue
        manifest = factory.get_manifest()
        runtime = registry.get_runtime(name)

        # Determine plugin type via factory helper
        from ccproxy.core.plugins import factory_type_name

        plugin_type = factory_type_name(factory)

        entries.append(
            PluginStatusEntry(
                name=name,
                version=manifest.version,
                type=plugin_type,
                provides=list(manifest.provides),
                requires=list(manifest.requires),
                optional_requires=list(manifest.optional_requires),
                initialized=runtime is not None
                and getattr(runtime, "initialized", False),
            )
        )

    # Extract init order and services map
    init_order = list(getattr(registry, "initialization_order", []) or [])
    services_map = dict(getattr(registry, "_service_providers", {}) or {})

    return PluginStatusResponse(
        initialization_order=init_order,
        services=services_map,
        plugins=entries,
    )


@router.delete("/{plugin_name}")
async def unregister_plugin() -> dict[str, str]:
    """Plugin unregistration is not supported in v2; endpoint removed."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Plugin unregistration is not supported; restart with desired config.",
    )
