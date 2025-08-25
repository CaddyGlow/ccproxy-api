"""Plugin management API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from starlette import status

from ccproxy.auth.conditional import ConditionalAuthDep


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


class PluginHealthResponse(BaseModel):
    """Response model for plugin health check."""

    plugin: str
    status: str  # "healthy", "unhealthy", "unknown"
    adapter_loaded: bool
    details: dict[str, Any] | None = None


class PluginReloadResponse(BaseModel):
    """Response model for plugin reload."""

    status: str  # "success", "error"
    message: str
    plugin: str | None = None


# Note: get_proxy_service dependency removed as v2 plugins don't use ProxyService for plugin management
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
        from ccproxy.plugins.factory import PluginRegistry

        registry: PluginRegistry = request.app.state.plugin_registry

        for name in registry.list_plugins():
            factory = registry.get_factory(name)
            if factory:
                manifest = factory.get_manifest()
                plugins.append(
                    PluginInfo(
                        name=name,
                        type="plugin",
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

    from ccproxy.plugins.factory import PluginRegistry

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
                status_value = getattr(health_result, "status", "unknown")  # type: ignore[unreachable]
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
            import structlog

            logger = structlog.get_logger(__name__)
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
            import structlog

            logger = structlog.get_logger(__name__)
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


@router.post("/{plugin_name}/reload", response_model=PluginReloadResponse)
async def reload_plugin(
    plugin_name: str,
    request: Request,
    auth: ConditionalAuthDep = None,
) -> PluginReloadResponse:
    """Reload a specific plugin.

    Args:
        plugin_name: Name of the plugin to reload

    Returns:
        Reload status

    Raises:
        HTTPException: If plugin not found or reload fails
    """
    # Access v2 plugin registry from app state
    if not hasattr(request.app.state, "plugin_registry"):
        raise HTTPException(status_code=503, detail="Plugin registry not initialized")

    from ccproxy.plugins.factory import PluginRegistry

    registry: PluginRegistry = request.app.state.plugin_registry

    # Check if plugin exists
    if plugin_name not in registry.list_plugins():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{plugin_name}' not found",
        )

    # V2 plugins are discovered at startup and cannot be reloaded at runtime
    # They are loaded from the filesystem during app creation
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Plugin reloading is not supported in v2 plugin system. Restart the server to reload plugins.",
    )


@router.post("/discover", response_model=PluginListResponse)
async def discover_plugins(
    request: Request,
    auth: ConditionalAuthDep = None,
) -> PluginListResponse:
    """Re-discover plugins from the plugin directory.

    Note: In v2 plugin system, plugins are discovered at app startup.
    This endpoint returns the current list without re-discovery.

    Returns:
        Current list of all plugins
    """
    # V2 plugins are discovered during app creation and cannot be re-discovered at runtime
    # Return the current list of plugins
    return await list_plugins(request, auth)


@router.delete("/{plugin_name}")
async def unregister_plugin(
    plugin_name: str,
    request: Request,
    auth: ConditionalAuthDep = None,
) -> dict[str, str]:
    """Unregister a plugin.

    Args:
        plugin_name: Name of the plugin to unregister

    Returns:
        Status message

    Raises:
        HTTPException: If plugin not found
    """
    # Access v2 plugin registry from app state
    if not hasattr(request.app.state, "plugin_registry"):
        raise HTTPException(status_code=503, detail="Plugin registry not initialized")

    from ccproxy.plugins.factory import PluginRegistry

    registry: PluginRegistry = request.app.state.plugin_registry

    # Check if plugin exists
    if plugin_name not in registry.list_plugins():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Plugin '{plugin_name}' not found",
        )

    # Try to shutdown the plugin runtime
    runtime = registry.get_runtime(plugin_name)
    if runtime:
        try:
            await runtime.shutdown()
            # Remove runtime from registry's internal dict
            if plugin_name in registry.runtimes:
                del registry.runtimes[plugin_name]
            return {
                "status": "success",
                "message": f"Plugin '{plugin_name}' unregistered successfully",
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unregister plugin '{plugin_name}': {str(e)}",
            ) from e

    # Should not reach here if plugin exists
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to unregister plugin '{plugin_name}'",
    )
