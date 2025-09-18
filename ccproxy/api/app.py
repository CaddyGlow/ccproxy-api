"""FastAPI application factory for CCProxy API Server with plugin system."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.routing import APIRouter
from typing_extensions import TypedDict

from ccproxy.api.bootstrap import create_service_container
from ccproxy.api.middleware.cors import setup_cors_middleware
from ccproxy.api.middleware.errors import setup_error_handlers
from ccproxy.api.routes.health import router as health_router
from ccproxy.api.routes.plugins import router as plugins_router
from ccproxy.auth.oauth.router import oauth_router
from ccproxy.config.settings import Settings
from ccproxy.core import __version__
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.core.logging import TraceBoundLogger, get_logger, setup_logging
from ccproxy.core.plugins import (
    MiddlewareManager,
    PluginRegistry,
    load_plugin_system,
    setup_default_middleware,
)
from ccproxy.core.plugins.hooks import HookManager
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.services.container import ServiceContainer
from ccproxy.utils.startup_helpers import (
    check_claude_cli_startup,
    check_version_updates_startup,
    setup_scheduler_shutdown,
    setup_scheduler_startup,
)


logger: TraceBoundLogger = get_logger()


def merge_router_tags(
    router: APIRouter,
    spec_tags: list[str] | None = None,
    default_tags: list[str] | None = None,
) -> list[str | Enum] | None:
    """Merge router tags with spec tags, removing duplicates while preserving order.

    Args:
        router: FastAPI router instance
        spec_tags: Tags from route specification
        default_tags: Fallback tags if no other tags exist

    Returns:
        Deduplicated list of tags, or None if no tags
    """
    router_tags: list[str | Enum] = list(router.tags) if router.tags else []
    spec_tags_list: list[str | Enum] = list(spec_tags) if spec_tags else []
    default_tags_list: list[str | Enum] = list(default_tags) if default_tags else []

    # Only use defaults if no other tags exist
    if not router_tags and not spec_tags_list and default_tags_list:
        return default_tags_list

    # Merge all non-default tags and deduplicate
    all_tags: list[str | Enum] = router_tags + spec_tags_list
    if not all_tags:
        return None

    # Deduplicate by string value while preserving order
    unique: list[str | Enum] = []
    seen: set[str] = set()
    for t in all_tags:
        s = str(t)
        if s not in seen:
            seen.add(s)
            unique.append(t)
    return unique


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


async def setup_task_manager_startup(app: FastAPI, settings: Settings) -> None:
    """Start the async task manager."""
    await start_task_manager()
    logger.debug("task_manager_startup_completed", category="lifecycle")


async def setup_task_manager_shutdown(app: FastAPI) -> None:
    """Stop the async task manager."""
    await stop_task_manager()
    logger.debug("task_manager_shutdown_completed", category="lifecycle")


async def setup_service_container_shutdown(app: FastAPI) -> None:
    """Close the service container and its resources."""
    if hasattr(app.state, "service_container"):
        service_container = app.state.service_container
        await service_container.shutdown()


async def initialize_plugins_startup(app: FastAPI, settings: Settings) -> None:
    """Initialize plugins during startup (runtime phase)."""
    if not settings.enable_plugins:
        logger.info("plugin_system_disabled", category="lifecycle")
        return

    if not hasattr(app.state, "plugin_registry"):
        logger.warning("plugin_registry_not_found", category="lifecycle")
        return

    plugin_registry: PluginRegistry = app.state.plugin_registry
    service_container: ServiceContainer = app.state.service_container

    hook_registry = service_container.get_hook_registry()
    background_thread_manager = service_container.get_background_hook_thread_manager()
    hook_manager = HookManager(hook_registry, background_thread_manager)
    app.state.hook_registry = hook_registry
    app.state.hook_manager = hook_manager
    service_container.register_service(HookManager, instance=hook_manager)

    # StreamingHandler now requires HookManager at construction via DI factory,
    # so no post-hoc patching is needed here.

    class CoreServicesAdapter:
        def __init__(self, container: ServiceContainer):
            self.settings = container.settings
            self.http_pool_manager = container.get_pool_manager()
            self.logger = get_logger()
            self.cli_detection_service = container.get_cli_detection_service()
            self.scheduler = getattr(app.state, "scheduler", None)
            self.plugin_registry = app.state.plugin_registry
            self.oauth_registry = getattr(app.state, "oauth_registry", None)
            self._container = container
            self.hook_registry = getattr(app.state, "hook_registry", None)
            self.hook_manager = getattr(app.state, "hook_manager", None)
            self.app = app
            self.request_tracer = container.get_request_tracer()
            self.streaming_handler = container.get_streaming_handler()
            self.metrics = None
            self.format_registry = container.get_format_registry()
            self.formatter_registry = container.get_formatter_registry()

        def get_plugin_config(self, plugin_name: str) -> Any:
            if hasattr(self.settings, "plugins") and self.settings.plugins:
                plugin_config = self.settings.plugins.get(plugin_name)
                if plugin_config:
                    return (
                        plugin_config.model_dump()
                        if hasattr(plugin_config, "model_dump")
                        else plugin_config
                    )
            return {}

        def get_format_registry(self) -> Any:
            """Get format adapter registry service instance."""
            return self.format_registry

        def get_formatter_registry(self) -> Any:
            """Get formatter registry service instance."""
            return self.formatter_registry

    core_services = CoreServicesAdapter(service_container)

    # Perform manifest population with access to http_pool_manager
    # This allows plugins to modify their manifests during context creation
    for plugin_name, factory in plugin_registry.factories.items():
        try:
            factory.create_context(core_services)
        except Exception as e:
            logger.warning(
                "plugin_context_creation_failed",
                plugin=plugin_name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )
            # Continue with other plugins

    from typing import cast as _cast

    from ccproxy.core.services import CoreServices as _CoreServices

    await plugin_registry.initialize_all(_cast(_CoreServices, core_services))

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


async def shutdown_hook_system(app: FastAPI) -> None:
    """Shutdown the hook system and background thread."""
    try:
        # Get hook manager from app state - it will shutdown its own background manager
        hook_manager = getattr(app.state, "hook_manager", None)
        if hook_manager:
            hook_manager.shutdown()

        logger.debug("hook_system_shutdown_completed", category="lifecycle")
    except Exception as e:
        logger.error(
            "hook_system_shutdown_failed",
            error=str(e),
            category="lifecycle",
        )


async def initialize_hooks_startup(app: FastAPI, settings: Settings) -> None:
    """Initialize hook system with plugins."""
    if hasattr(app.state, "hook_registry") and hasattr(app.state, "hook_manager"):
        hook_registry = app.state.hook_registry
        hook_manager = app.state.hook_manager
        logger.debug("hook_system_already_created", category="lifecycle")
    else:
        service_container: ServiceContainer = app.state.service_container
        hook_registry = service_container.get_hook_registry()
        background_thread_manager = (
            service_container.get_background_hook_thread_manager()
        )
        hook_manager = HookManager(hook_registry, background_thread_manager)
        app.state.hook_registry = hook_registry
        app.state.hook_manager = hook_manager

    # Register core HTTP tracer hook first (high priority)
    try:
        from ccproxy.core.plugins.hooks.implementations import HTTPTracerHook
        from ccproxy.core.plugins.hooks.implementations.formatters import (
            JSONFormatter,
            RawHTTPFormatter,
        )

        # Check if core HTTP tracing should be enabled
        # We'll enable it if logging.enable_plugin_logging is True and no explicit disable is set
        core_tracer_enabled = getattr(settings.logging, "enable_plugin_logging", True)

        if core_tracer_enabled:
            # Create formatters with settings-based configuration
            log_dir = getattr(settings.logging, "plugin_log_base_dir", "/tmp/ccproxy")

            json_formatter = JSONFormatter(
                log_dir=f"{log_dir}/tracer",
                verbose_api=getattr(settings.logging, "verbose_api", True),
                json_logs_enabled=True,
                redact_sensitive=True,
                truncate_body_preview=1024,
            )

            raw_formatter = RawHTTPFormatter(
                log_dir=f"{log_dir}/tracer",
                enabled=True,
                log_client_request=True,
                log_client_response=True,
                log_provider_request=True,
                log_provider_response=True,
                max_body_size=10485760,  # 10MB
                exclude_headers=[
                    "authorization",
                    "x-api-key",
                    "cookie",
                    "x-auth-token",
                ],
            )

            # Create and register core HTTP tracer
            core_http_tracer = HTTPTracerHook(
                json_formatter=json_formatter,
                raw_formatter=raw_formatter,
                enabled=True,
            )

            hook_registry.register(core_http_tracer)
            logger.info(
                "core_http_tracer_registered",
                hook_name=core_http_tracer.name,
                events=core_http_tracer.events,
                category="lifecycle",
            )
        else:
            logger.debug(
                "core_http_tracer_disabled",
                reason="plugin_logging_disabled",
                category="lifecycle",
            )

    except Exception as e:
        logger.error(
            "core_http_tracer_registration_failed",
            error=str(e),
            exc_info=e,
            category="lifecycle",
        )

    # Register plugin hooks
    if hasattr(app.state, "plugin_registry"):
        plugin_registry: PluginRegistry = app.state.plugin_registry

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

    try:
        await hook_manager.emit(HookEvent.APP_STARTUP, {"phase": "startup"})
    except Exception as e:
        logger.error(
            "startup_hook_failed", error=str(e), exc_info=e, category="lifecycle"
        )

    logger.info(
        "hook_system_initialized",
        hook_count=len(hook_registry._hooks),
        category="lifecycle",
    )


LIFECYCLE_COMPONENTS: list[LifecycleComponent] = [
    {
        "name": "Task Manager",
        "startup": setup_task_manager_startup,
        "shutdown": setup_task_manager_shutdown,
    },
    {
        "name": "Version Check",
        "startup": check_version_updates_startup,
        "shutdown": None,
    },
    {
        "name": "Claude CLI",
        "startup": check_claude_cli_startup,
        "shutdown": None,
    },
    {
        "name": "Scheduler",
        "startup": setup_scheduler_startup,
        "shutdown": setup_scheduler_shutdown,
    },
    {
        "name": "Service Container",
        "startup": None,
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
        "shutdown": shutdown_hook_system,
    },
]

SHUTDOWN_ONLY_COMPONENTS: list[ShutdownComponent] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager using component-based approach."""
    service_container: ServiceContainer = app.state.service_container
    settings = service_container.get_service(Settings)

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
            except Exception as e:
                logger.error(
                    f"{component_name.lower().replace(' ', '_')}_startup_failed",
                    error=str(e),
                    component=component_name,
                    exc_info=e,
                    category="lifecycle",
                )

    yield

    logger.debug("server_stop", category="lifecycle")

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

    for component in reversed(LIFECYCLE_COMPONENTS):
        if component["shutdown"]:
            component_name = component["name"]
            try:
                logger.debug(
                    f"stopping_{component_name.lower().replace(' ', '_')}",
                    category="lifecycle",
                )
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


def create_app(service_container: ServiceContainer | None = None) -> FastAPI:
    if service_container is None:
        service_container = create_service_container()
    """Create and configure the FastAPI application with plugin system."""
    settings = service_container.get_service(Settings)
    if not structlog.is_configured():
        json_logs = settings.logging.format == "json"

        logger.error(
            "structlog_not_configured", category="lifecycle", settings=settings
        )
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

    app.state.service_container = service_container

    app.state.oauth_registry = service_container.get_oauth_registry()

    plugin_registry = PluginRegistry()
    middleware_manager = MiddlewareManager()

    if settings.enable_plugins:
        plugin_registry, middleware_manager = load_plugin_system(settings)

        logger.info(
            "plugins_registered",
            total=len(plugin_registry.factories),
            providers=sum(
                1
                for f in plugin_registry.factories.values()
                if f.get_manifest().is_provider
            ),
            system_plugins=len(plugin_registry.factories)
            - sum(
                1
                for f in plugin_registry.factories.values()
                if f.get_manifest().is_provider
            ),
            names=list(plugin_registry.factories.keys()),
            category="plugin",
        )

        # Manifest population will be done during startup when core services are available

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

        for name, factory in plugin_registry.factories.items():
            manifest = factory.get_manifest()
            for route_spec in manifest.routes:
                default_tag = name.replace("_", "-")
                # Merge router tags with spec tags, removing duplicates
                merged_tags = merge_router_tags(
                    route_spec.router,
                    spec_tags=route_spec.tags,
                    default_tags=[default_tag],
                )

                app.include_router(
                    route_spec.router,
                    prefix=route_spec.prefix,
                    tags=merged_tags,
                    dependencies=route_spec.dependencies,
                )
                logger.debug(
                    "plugin_routes_registered",
                    plugin=name,
                    prefix=route_spec.prefix,
                    category="lifecycle",
                )

    app.state.plugin_registry = plugin_registry
    app.state.middleware_manager = middleware_manager

    app.state.settings = settings

    setup_cors_middleware(app, settings)
    setup_error_handlers(app)

    # Add format chain middleware
    from ccproxy.api.middleware.format_chain import FormatChainMiddleware

    app.add_middleware(FormatChainMiddleware)

    # Register core converters into the format registry and validate route chains
    try:
        from ccproxy.services.adapters.chain_validation import validate_chains
        from ccproxy.services.adapters.simple_converters import (
            register_converters,
        )

        registry = service_container.get_format_registry()
        register_converters(registry, plugin_name="core")

        # Collect declared chains from routes for validation
        declared_chains: list[list[str]] = []
        for route in app.router.routes:  # type: ignore[attr-defined]
            endpoint = getattr(route, "endpoint", None)
            chain = getattr(endpoint, "__format_chain__", None)
            if chain:
                declared_chains.append(chain)

        missing = validate_chains(registry=registry, chains=declared_chains)
        if missing:
            logger.error("format_chain_validation_failed", missing_adapters=missing)
    except Exception as _e:
        # Best‑effort registration/validation; do not block app startup
        logger.warning("format_registry_setup_skipped", error=str(_e))

    setup_default_middleware(middleware_manager)

    middleware_manager.apply_to_app(app)

    # Core router registrations with tag merging
    app.include_router(
        health_router, tags=merge_router_tags(health_router, default_tags=["health"])
    )

    app.include_router(
        oauth_router,
        prefix="/oauth",
        tags=merge_router_tags(oauth_router, default_tags=["oauth"]),
    )

    if settings.enable_plugins:
        app.include_router(
            plugins_router,
            tags=merge_router_tags(plugins_router, default_tags=["plugins"]),
        )

    return app


def get_app() -> FastAPI:
    """Get the FastAPI app instance."""
    container = create_service_container()
    return create_app(container)


__all__ = ["create_app", "get_app"]
