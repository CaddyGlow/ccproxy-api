"""FastAPI application factory for CCProxy API Server with plugin system."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from typing_extensions import TypedDict

from ccproxy import __version__
from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.plugins import router as plugins_router
from ccproxy.auth.oauth.registry import OAuthRegistry
from ccproxy.auth.oauth.routes import router as oauth_router
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.core.logging import TraceBoundLogger, get_logger, setup_logging
from ccproxy.hooks import HookManager, HookRegistry
from ccproxy.hooks.events import HookEvent

# Plugin System imports
from ccproxy.plugins import (
    MiddlewareManager,
    PluginRegistry,
    discover_and_load_plugins,
    setup_default_middleware,
)
from ccproxy.services.container import ServiceContainer
from ccproxy.services.factories import ConcreteServiceFactory
from ccproxy.utils.startup_helpers import (
    check_claude_cli_startup,
    check_version_updates_startup,
    initialize_service_container_startup,
    setup_scheduler_shutdown,
    setup_scheduler_startup,
)


logger: TraceBoundLogger = get_logger()


# Type definitions for lifecycle components
class LifecycleComponent(TypedDict):
    name: str
    startup: Callable[[FastAPI, Any], Awaitable[None]] | None
    shutdown: (
        Callable[[FastAPI], Awaitable[None]]
        | Callable[[FastAPI, Any], Awaitable[None]]
        | None
    )


class ShutdownComponent(TypedDict):
    name: str
    shutdown: Callable[[FastAPI], Awaitable[None]] | None


# Define startup/shutdown functions first
async def setup_task_manager_startup(app: FastAPI, settings: Settings) -> None:
    """Start the async task manager."""
    await start_task_manager()
    logger.debug("task_manager_startup_completed", category="lifecycle")


async def setup_task_manager_shutdown(app: FastAPI) -> None:
    """Stop the async task manager."""
    await stop_task_manager()
    logger.debug("task_manager_shutdown_completed", category="lifecycle")


# Legacy HTTP client singleton shutdown removed; HTTP client is managed by ServiceContainer


async def setup_service_container_shutdown(app: FastAPI) -> None:
    """Close the service container and its resources."""
    if hasattr(app.state, "service_container"):
        service_container = app.state.service_container
        if hasattr(service_container, "close"):
            try:
                await service_container.close()
                logger.debug(
                    "service_container_shutdown_completed", category="lifecycle"
                )
            except Exception as e:
                logger.error(
                    "service_container_shutdown_failed",
                    error=str(e),
                    exc_info=e,
                    category="lifecycle",
                )


async def initialize_plugins_startup(app: FastAPI, settings: Settings) -> None:
    """Initialize plugins during startup (runtime phase)."""
    if not settings.enable_plugins:
        logger.info("plugin_system_disabled", category="lifecycle")
        return

    # Get plugin registry from app state (set during app creation)
    if not hasattr(app.state, "plugin_registry"):
        logger.warning("plugin_registry_not_found", category="lifecycle")
        return

    plugin_registry: PluginRegistry = app.state.plugin_registry

    # Create service container for plugin initialization
    if not hasattr(app.state, "service_container"):
        factory = ConcreteServiceFactory()
        app.state.service_container = ServiceContainer(settings, factory)

    service_container = app.state.service_container

    # Create hook system early if enabled so plugins can register hooks
    hook_registry = None
    hook_manager = None
    if settings.hooks.enabled:
        hook_registry = HookRegistry()
        hook_manager = HookManager(hook_registry)
        # Store in app state so plugins can access during initialization
        app.state.hook_registry = hook_registry
        app.state.hook_manager = hook_manager
        logger.debug("hook_system_created_early_for_plugins", category="lifecycle")

    # Create a core services adapter for the plugin system
    # The plugin system expects certain attributes that we need to provide
    class CoreServicesAdapter:
        def __init__(self, container: ServiceContainer):
            self.settings = container.settings
            self.http_client = container.get_http_client()
            self.logger = get_logger()
            self.cli_detection_service = container.get_cli_detection_service()
            self.scheduler = getattr(app.state, "scheduler", None)
            self.plugin_registry = app.state.plugin_registry
            self.oauth_registry = getattr(app.state, "oauth_registry", None)
            self._container = container
            # Add hook registry and manager if available
            self.hook_registry = getattr(app.state, "hook_registry", None)
            self.hook_manager = getattr(app.state, "hook_manager", None)
            # Also expose app for plugins that need app.state access
            self.app = app
            # Add streaming handler and other services
            self.request_tracer = container.get_request_tracer()
            # Pass hook_manager to streaming handler for event emission
            self.streaming_handler = container.get_streaming_handler(
                hook_manager=self.hook_manager
            )
            # Metrics is now handled by the metrics plugin, not core services
            self.metrics = None

        def get_plugin_config(self, plugin_name: str) -> Any:
            """Get plugin configuration."""
            # Check if plugin config exists in settings
            if hasattr(self.settings, "plugins") and self.settings.plugins:
                plugin_config = self.settings.plugins.get(plugin_name)
                if plugin_config:
                    return (
                        plugin_config.model_dump()
                        if hasattr(plugin_config, "model_dump")
                        else plugin_config
                    )

            # Return empty config as default
            return {}

    core_services = CoreServicesAdapter(service_container)

    # Initialize all plugins with their runtime context
    await plugin_registry.initialize_all(core_services)

    # No plugin-specific routing logic in core. Plugins register their own routes.

    logger.info(
        "plugins_initialization_completed",
        total_plugins=len(plugin_registry.list_plugins()),
        provider_plugins=len(plugin_registry.list_provider_plugins()),
        category="lifecycle",
    )


async def shutdown_plugins(app: FastAPI) -> None:
    """Shutdown plugins."""
    if hasattr(app.state, "plugin_registry"):
        plugin_registry: PluginRegistry = app.state.plugin_registry
        await plugin_registry.shutdown_all()
        logger.debug("plugins_shutdown_completed", category="lifecycle")


async def initialize_hooks_startup(app: FastAPI, settings: Settings) -> None:
    """Initialize hook system with plugins."""
    if not settings.hooks.enabled:
        logger.info("hook_system_disabled", category="lifecycle")
        return

    # Check if hook system was already created during plugin init
    if hasattr(app.state, "hook_registry") and hasattr(app.state, "hook_manager"):
        hook_registry = app.state.hook_registry
        hook_manager = app.state.hook_manager
        logger.debug("hook_system_already_created", category="lifecycle")
    else:
        # Create hook system if not already created
        hook_registry = HookRegistry()
        hook_manager = HookManager(hook_registry)
        # Store hook manager in app state
        app.state.hook_registry = hook_registry
        app.state.hook_manager = hook_manager

    # Get plugin registry from app state
    if hasattr(app.state, "plugin_registry"):
        plugin_registry: PluginRegistry = app.state.plugin_registry

        # Register hooks from plugin manifests
        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            for hook_spec in manifest.hooks:
                try:
                    hook_instance = hook_spec.hook_class(**hook_spec.kwargs)
                    hook_registry.register(hook_instance)
                    logger.debug(
                        "plugin_hook_registered",
                        plugin_name=name,
                        hook_class=hook_spec.hook_class.__name__,
                        category="lifecycle",
                    )
                except Exception as e:
                    logger.error(
                        "plugin_hook_registration_failed",
                        plugin_name=name,
                        hook_class=hook_spec.hook_class.__name__,
                        error=str(e),
                        exc_info=e,
                        category="lifecycle",
                    )

    # Hooks middleware is already added during middleware setup and will get
    # the hook manager from app state when processing requests

    # Trigger startup hook
    try:
        # Use the APP_STARTUP event from the enum
        await hook_manager.emit(HookEvent.APP_STARTUP, {"phase": "startup"})
    except Exception as e:
        logger.error(
            "startup_hook_failed", error=str(e), exc_info=e, category="lifecycle"
        )

    # Use _hooks to get the count (or better, add a method to get count)
    logger.info(
        "hook_system_initialized",
        hook_count=len(hook_registry._hooks),
        category="lifecycle",
    )


# Define lifecycle components in order
LIFECYCLE_COMPONENTS: list[LifecycleComponent] = [
    {
        "name": "Task Manager",
        "startup": setup_task_manager_startup,
        "shutdown": setup_task_manager_shutdown,
    },
    {
        "name": "Version Check",
        "startup": check_version_updates_startup,
        "shutdown": None,  # One-time check, no cleanup needed
    },
    {
        "name": "Claude CLI",
        "startup": check_claude_cli_startup,
        "shutdown": None,  # Detection only, no cleanup needed
    },
    {
        "name": "Scheduler",
        "startup": setup_scheduler_startup,
        "shutdown": setup_scheduler_shutdown,
    },
    # Log storage is initialized by the duckdb_storage plugin.
    {
        "name": "Service Container",
        "startup": initialize_service_container_startup,
        "shutdown": setup_service_container_shutdown,
    },
    {
        "name": "Plugin System",
        "startup": initialize_plugins_startup,
        "shutdown": shutdown_plugins,
    },
    {
        "name": "Hook System",
        "startup": initialize_hooks_startup,
        "shutdown": None,  # Hook system cleaned up automatically
    },
]

SHUTDOWN_ONLY_COMPONENTS: list[ShutdownComponent] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager using component-based approach."""
    # Use settings from app.state if available (modified by plugins during create_app)
    # Otherwise get fresh settings
    if hasattr(app.state, "settings"):
        settings = app.state.settings
    else:
        settings = get_settings()
        # Store settings in app state for reuse in dependencies
        app.state.settings = settings

    # Startup
    logger.info(
        "server_start",
        host=settings.server.host,
        port=settings.server.port,
        url=f"http://{settings.server.host}:{settings.server.port}",
        category="lifecycle",
    )
    logger.debug(
        "server_configured",
        host=settings.server.host,
        port=settings.server.port,
        category="config",
    )

    # Execute startup components in order
    for component in LIFECYCLE_COMPONENTS:
        if component["startup"]:
            component_name = component["name"]
            try:
                logger.debug(
                    f"starting_{component_name.lower().replace(' ', '_')}",
                    category="lifecycle",
                )
                await component["startup"](app, settings)
            except (OSError, PermissionError) as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_startup_io_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )
                # Continue with graceful degradation
            except Exception as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_startup_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )
                # Continue with graceful degradation

    yield

    # Shutdown
    logger.debug("server_stop", category="lifecycle")

    # Execute shutdown-only components first
    for shutdown_component in SHUTDOWN_ONLY_COMPONENTS:
        if shutdown_component["shutdown"]:
            component_name = shutdown_component["name"]
            try:
                logger.debug(
                    f"stopping_{component_name.lower().replace(' ', '_')}",
                    category="lifecycle",
                )
                await shutdown_component["shutdown"](app)
            except (OSError, PermissionError) as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_shutdown_io_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )
            except Exception as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_shutdown_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )

    # Execute shutdown components in reverse order
    for component in reversed(LIFECYCLE_COMPONENTS):
        if component["shutdown"]:
            component_name = component["name"]
            try:
                logger.debug(
                    f"stopping_{component_name.lower().replace(' ', '_')}",
                    category="lifecycle",
                )
                # Some shutdown functions need settings, others don't
                if component_name == "Permission Service":
                    await component["shutdown"](app, settings)  # type: ignore
                else:
                    await component["shutdown"](app)  # type: ignore
            except (OSError, PermissionError) as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_shutdown_io_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )
            except Exception as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_shutdown_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application with plugin system.

    Args:
        settings: Optional settings override. If None, uses get_settings().

    Returns:
        Configured FastAPI application instance.
    """
    if settings is None:
        settings = get_settings()

    # Configure logging based on settings BEFORE any module uses logger

    if not structlog.is_configured():
        # Only setup logging if structlog is not configured at all
        # Use JSON logs based on the logging format setting
        json_logs = settings.logging.format == "json"
        setup_logging(
            json_logs=json_logs,
            log_level_name=settings.logging.level,
            log_file=settings.logging.file,
        )

    app = FastAPI(
        title="CCProxy API Server",
        description="High-performance API server providing Anthropic and OpenAI-compatible interfaces for Claude AI models",
        version=__version__,
        lifespan=lifespan,
    )

    # Create core-scoped OAuth registry (avoid module-level globals)
    app.state.oauth_registry = OAuthRegistry()

    # PHASE 1: Plugin Discovery and Registration (before app starts)
    plugin_registry = PluginRegistry()
    middleware_manager = MiddlewareManager()

    if settings.enable_plugins:
        # Discover and load plugin factories
        plugin_factories = discover_and_load_plugins(settings)

        # Register all plugin factories
        for factory in plugin_factories.values():
            plugin_registry.register_factory(factory)

        # Log registration summary
        provider_count = sum(
            1 for f in plugin_factories.values() if f.get_manifest().is_provider
        )
        logger.info(
            "plugins_registered",
            total=len(plugin_factories),
            providers=provider_count,
            system_plugins=len(plugin_factories) - provider_count,
            names=list(plugin_factories.keys()),
            category="plugin",
        )

        # Create a minimal core services adapter for manifest population
        # This allows plugins to check their configuration and add middleware/routes
        class ManifestPopulationServices:
            def __init__(self, settings: Settings | None) -> None:
                self.settings = settings
                self.http_client = None  # Not needed for manifest population
                self.logger = structlog.get_logger()

            def get_plugin_config(self, plugin_name: str) -> dict[str, Any]:
                # Use the settings.plugins dictionary populated by Pydantic
                if (
                    self.settings
                    and hasattr(self.settings, "plugins")
                    and self.settings.plugins
                ):
                    return self.settings.plugins.get(plugin_name, {})
                return {}

        manifest_services = ManifestPopulationServices(settings)

        # Call create_context on each factory to populate manifests
        # This allows plugins to conditionally add middleware/routes based on config
        for _name, factory in plugin_registry.factories.items():
            factory.create_context(manifest_services)

        # Collect middleware from all plugins
        plugin_middleware_count = 0
        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            if manifest.middleware:
                middleware_manager.add_plugin_middleware(name, manifest.middleware)
                plugin_middleware_count += len(manifest.middleware)
                logger.trace(
                    "plugin_middleware_collected",
                    plugin=name,
                    count=len(manifest.middleware),
                    category="lifecycle",
                )

        # Log middleware collection summary
        if plugin_middleware_count > 0:
            plugins_with_middleware = [
                n
                for n, f in plugin_registry.factories.items()
                if f.get_manifest().middleware
            ]
            logger.debug(
                "plugin_middleware_collection_completed",
                total_middleware=plugin_middleware_count,
                plugins_with_middleware=len(plugins_with_middleware),
                plugin_names=plugins_with_middleware,
                category="lifecycle",
            )

        # Register plugin routes (static registration during app creation)
        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            for route_spec in manifest.routes:
                app.include_router(
                    route_spec.router,
                    prefix=route_spec.prefix,
                    tags=list(route_spec.tags)
                    if route_spec.tags
                    else [f"plugin-{name}"],
                    dependencies=route_spec.dependencies,
                )
                logger.debug(
                    "plugin_routes_registered",
                    plugin=name,
                    prefix=route_spec.prefix,
                    category="lifecycle",
                )

    # Store plugin registry and middleware manager in app state for runtime initialization
    app.state.plugin_registry = plugin_registry
    app.state.middleware_manager = middleware_manager

    # Store settings object that was potentially modified by plugins
    # This ensures the same settings object is used throughout the app lifecycle
    app.state.settings = settings

    # Setup CORS middleware first (needs to be outermost)
    setup_cors_middleware(app, settings)
    setup_error_handlers(app)

    # Setup default core middleware
    setup_default_middleware(middleware_manager)

    # Hooks middleware will be added during hook system initialization

    # Apply all middleware in correct order
    middleware_manager.apply_to_app(app)

    # Include core routers
    app.include_router(health_router, tags=["health"])

    # Metrics endpoint is provided by the metrics plugin when enabled.

    # Logs and dashboard routes are provided by plugins: analytics and dashboard.

    app.include_router(oauth_router, prefix="/oauth", tags=["oauth"])

    # Plugin management endpoints (conditional on plugin system)
    if settings.enable_plugins:
        app.include_router(plugins_router, prefix="/api", tags=["plugins"])

    # Dashboard static mounting is handled by the dashboard plugin.

    return app


def get_app() -> FastAPI:
    """Get the FastAPI app instance.

    This is a convenience function for backwards compatibility.
    """
    from ccproxy.config.settings import get_settings

    return create_app(get_settings())


# Export create_app as the main factory
__all__ = ["create_app", "get_app"]
