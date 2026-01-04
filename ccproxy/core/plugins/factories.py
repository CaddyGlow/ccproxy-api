"""Plugin factory implementations and registry.

This module contains all concrete factory implementations merged from
base_factory.py and factory.py to eliminate circular dependencies.
"""

import inspect
from typing import TYPE_CHECKING, Any, cast

import httpx
import structlog
from fastapi import APIRouter

from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.interfaces import (
    IMetricsCollector,
    IRequestTracer,
    NullMetricsCollector,
    NullRequestTracer,
    NullStreamingHandler,
    StreamingMetrics,
)
from ccproxy.utils.model_mapper import ModelMapper

from .declaration import (
    CliArgumentSpec,
    CliCommandSpec,
    FormatAdapterSpec,
    FormatPair,
    PluginContext,
    PluginManifest,
    RouterSpec,
    RouteSpec,
    TaskSpec,
)
from .interfaces import (
    AuthProviderPluginFactory,
    PluginFactory,
    ProviderPluginFactory,
)


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings
    from ccproxy.http.pool import HTTPPoolManager
    from ccproxy.services.container import ServiceContainer


logger = structlog.get_logger(__name__)

# Type variable for service type checking
T = Any


class BaseProviderPluginFactory(ProviderPluginFactory):
    """Base factory for provider plugins that eliminates common boilerplate.

    This class uses class attributes for plugin configuration and implements
    common methods that all provider factories share. Subclasses only need
    to define class attributes and override methods that need custom behavior.

    Required class attributes to be defined by subclasses:
    - plugin_name: str
    - plugin_description: str
    - runtime_class: type[ProviderPluginRuntime]
    - adapter_class: type[BaseAdapter]
    - config_class: type[BaseSettings]

    Optional class attributes with defaults:
    - plugin_version: str = "1.0.0"
    - detection_service_class: type | None = None
    - credentials_manager_class: type | None = None
    - router: APIRouter | None = None
    - route_prefix: str = "/api"
    - dependencies: list[str] = []
    - optional_requires: list[str] = []
    - tasks: list[TaskSpec] = []
    """

    # Required class attributes (must be overridden by subclasses)
    plugin_name: str
    plugin_description: str
    runtime_class: Any  # Should be type[ProviderPluginRuntime] subclass
    adapter_class: Any  # Should be type[BaseAdapter] subclass
    config_class: Any  # Should be type[BaseSettings] subclass

    # Optional class attributes with defaults
    plugin_version: str = "1.0.0"
    detection_service_class: type | None = None
    credentials_manager_class: type[Any] | None = None
    auth_manager_name: str | None = None  # String-based auth manager reference
    routers: list[RouterSpec] = []
    dependencies: list[str] = []
    optional_requires: list[str] = []
    tasks: list[TaskSpec] = []

    # Format adapter declarations (populated by subclasses)
    format_adapters: list[FormatAdapterSpec] = []
    requires_format_adapters: list[FormatPair] = []

    # CLI extension declarations (populated by subclasses)
    cli_commands: list[CliCommandSpec] = []
    cli_arguments: list[CliArgumentSpec] = []
    tool_accumulator_class: type | None = None

    def __init__(self) -> None:
        """Initialize factory with manifest built from class attributes."""
        # Validate required class attributes
        self._validate_class_attributes()

        # Validate runtime class is a proper subclass
        # Import locally to avoid circular import during module import
        from .runtime import ProviderPluginRuntime

        if not issubclass(self.runtime_class, ProviderPluginRuntime):
            raise TypeError(
                f"runtime_class {self.runtime_class.__name__} must be a subclass of ProviderPluginRuntime"
            )

        # Build routes from routers list
        routes = []
        for router_spec in self.routers:
            # Handle both router instances and router factory functions
            router_instance = router_spec.router
            if callable(router_spec.router) and not isinstance(
                router_spec.router, APIRouter
            ):
                # Router is a factory function, call it to get the actual router
                router_instance = router_spec.router()

            routes.append(
                RouteSpec(
                    router=cast(APIRouter, router_instance),
                    prefix=router_spec.prefix,
                    tags=router_spec.tags or [],
                    dependencies=router_spec.dependencies,
                )
            )

        # Create manifest from class attributes
        manifest = PluginManifest(
            name=self.plugin_name,
            version=self.plugin_version,
            description=self.plugin_description,
            is_provider=True,
            config_class=self.config_class,
            tool_accumulator_class=self.tool_accumulator_class,
            dependencies=self.dependencies.copy(),
            optional_requires=self.optional_requires.copy(),
            routes=routes,
            tasks=self.tasks.copy(),
            format_adapters=self.format_adapters.copy(),
            requires_format_adapters=self.requires_format_adapters.copy(),
            cli_commands=self.cli_commands.copy(),
            cli_arguments=self.cli_arguments.copy(),
        )

        # Format adapter specification validation is deferred to runtime
        # when settings are available via dependency injection

        # Store the manifest and runtime class directly
        # We don't call parent __init__ because ProviderPluginFactory
        # would override our runtime_class with ProviderPluginRuntime
        self.manifest = manifest
        self.runtime_class = self.__class__.runtime_class

    def validate_format_adapters_with_settings(self, settings: "Settings") -> None:
        """Validate format adapter specifications (feature flags removed)."""
        self._validate_format_adapter_specs()

    def _validate_class_attributes(self) -> None:
        """Validate that required class attributes are defined."""
        required_attrs = [
            "plugin_name",
            "plugin_description",
            "runtime_class",
            "adapter_class",
            "config_class",
        ]

        for attr in required_attrs:
            if (
                not hasattr(self.__class__, attr)
                or getattr(self.__class__, attr) is None
            ):
                raise ValueError(
                    f"Class attribute '{attr}' must be defined in {self.__class__.__name__}"
                )

    def _validate_format_adapter_specs(self) -> None:
        """Validate format adapter specifications."""
        for spec in self.format_adapters:
            if not callable(spec.adapter_factory):
                raise ValueError(
                    f"Invalid adapter factory for {spec.from_format} -> {spec.to_format}: "
                    f"must be callable"
                ) from None

    def create_runtime(self) -> Any:
        """Create runtime instance using the configured runtime class."""
        return cast(Any, self.runtime_class(self.manifest))

    async def create_adapter(self, context: PluginContext) -> BaseAdapter:
        """Create adapter instance with explicit dependencies.

        This method extracts services from context and creates the adapter
        with explicit dependency injection. Subclasses can override this
        method if they need custom adapter creation logic.

        Args:
            context: Plugin context

        Returns:
            Adapter instance
        """
        # Extract services from context (one-time extraction)
        http_pool_manager: HTTPPoolManager | None = cast(
            "HTTPPoolManager | None", context.get("http_pool_manager")
        )
        request_tracer: IRequestTracer | None = context.get("request_tracer")
        metrics: IMetricsCollector | None = context.get("metrics")
        streaming_handler: StreamingMetrics | None = context.get("streaming_handler")
        hook_manager = context.get("hook_manager")

        # Get auth and detection services that may have been created by factory
        auth_manager = context.get("credentials_manager")
        detection_service = context.get("detection_service")

        # Get config if available
        config = context.get("config")

        # Get all adapter dependencies from service container
        service_container = context.get("service_container")
        if not service_container:
            raise RuntimeError("Service container is required for adapter services")

        # Get standardized adapter dependencies
        adapter_dependencies = service_container.get_adapter_dependencies(metrics)

        # Check if this is an HTTP-based adapter
        if issubclass(self.adapter_class, BaseHTTPAdapter):
            # HTTP adapters require http_pool_manager
            if not http_pool_manager:
                raise RuntimeError(
                    f"HTTP pool manager required for {self.adapter_class.__name__} but not available in context"
                )

            # Ensure config is provided for HTTP adapters
            if config is None and self.manifest.config_class:
                config = self.manifest.config_class()

            # Create HTTP adapter with explicit dependencies including format services
            init_params = inspect.signature(self.adapter_class.__init__).parameters
            adapter_kwargs: dict[str, Any] = {
                "config": config,
                "auth_manager": auth_manager,
                "detection_service": detection_service,
                "http_pool_manager": http_pool_manager,
                "request_tracer": request_tracer or NullRequestTracer(),
                "metrics": metrics or NullMetricsCollector(),
                "streaming_handler": streaming_handler or NullStreamingHandler(),
                "hook_manager": hook_manager,
                "format_registry": adapter_dependencies["format_registry"],
                "context": context,
                "model_mapper": context.get("model_mapper")
                if hasattr(context, "get")
                else None,
            }
            if self.tool_accumulator_class:
                adapter_kwargs["tool_accumulator_class"] = self.tool_accumulator_class

            return cast(BaseAdapter, self.adapter_class(**adapter_kwargs))
        else:
            # Non-HTTP adapters (like ClaudeSDK) have different dependencies
            # Build kwargs based on adapter class constructor signature
            non_http_adapter_kwargs: dict[str, Any] = {}

            # Get the adapter's __init__ signature
            sig = inspect.signature(self.adapter_class.__init__)
            params = sig.parameters

            # For non-HTTP adapters, create http_client from pool manager if needed
            client_for_non_http: httpx.AsyncClient | None = None
            if http_pool_manager and "http_client" in params:
                client_for_non_http = await http_pool_manager.get_client()

            # Map available services to expected parameters
            param_mapping = {
                "config": config,
                "http_client": client_for_non_http,
                "http_pool_manager": http_pool_manager,
                "auth_manager": auth_manager,
                "detection_service": detection_service,
                "session_manager": context.get("session_manager"),
                "request_tracer": request_tracer,
                "metrics": metrics,
                "streaming_handler": streaming_handler,
                "hook_manager": hook_manager,
                "format_registry": adapter_dependencies["format_registry"],
                "context": context,
                "model_mapper": context.get("model_mapper")
                if hasattr(context, "get")
                else None,
            }
            if self.tool_accumulator_class:
                non_http_adapter_kwargs["tool_accumulator_class"] = (
                    self.tool_accumulator_class
                )

            # Add parameters that the adapter expects
            for param_name, param in params.items():
                if param_name in ("self", "kwargs"):
                    continue
                if param_name in param_mapping:
                    if param_mapping[param_name] is not None:
                        non_http_adapter_kwargs[param_name] = param_mapping[param_name]
                    elif (
                        param_name == "config"
                        and param.default is inspect.Parameter.empty
                        and self.manifest.config_class
                    ):
                        # Config is None but required, create default
                        default_config = self.manifest.config_class()
                        non_http_adapter_kwargs["config"] = default_config
                elif (
                    param.default is inspect.Parameter.empty
                    and param_name not in non_http_adapter_kwargs
                    and param_name == "config"
                    and self.manifest.config_class
                ):
                    # Config parameter is missing but required, create default
                    default_config = self.manifest.config_class()
                    non_http_adapter_kwargs["config"] = default_config

            return cast(BaseAdapter, self.adapter_class(**non_http_adapter_kwargs))

    def create_detection_service(self, context: PluginContext) -> Any:
        """Create detection service instance if class is configured.

        Args:
            context: Plugin context

        Returns:
            Detection service instance or None if no class configured
        """
        if self.detection_service_class is None:
            return None

        settings = context.get("settings")
        if settings is None:
            from ccproxy.config.settings import Settings

            settings = Settings()

        cli_service = context.get("cli_detection_service")
        return self.detection_service_class(settings, cli_service)

    async def create_credentials_manager(self, context: PluginContext) -> Any:
        """Resolve credentials manager via the shared auth registry."""

        auth_manager_name = self.get_auth_manager_name(context)
        registry = None

        service_container = context.get("service_container")
        if service_container and hasattr(
            service_container, "get_auth_manager_registry"
        ):
            registry = service_container.get_auth_manager_registry()

        if not auth_manager_name:
            return None

        if not registry:
            logger.warning(
                "auth_manager_registry_unavailable",
                plugin=self.manifest.name,
                auth_manager_name=auth_manager_name,
                category="auth",
            )
            return None

        resolved = await registry.get(auth_manager_name)
        if resolved:
            return resolved

        # Respect explicit overrides that could not be resolved
        if self.auth_manager_name and auth_manager_name != self.auth_manager_name:
            logger.warning(
                "auth_manager_override_not_resolved",
                plugin=self.manifest.name,
                auth_manager_name=auth_manager_name,
                category="auth",
            )
        else:
            logger.warning(
                "auth_manager_not_registered",
                plugin=self.manifest.name,
                auth_manager_name=auth_manager_name,
                category="auth",
            )

        return None

    def get_auth_manager_name(self, context: PluginContext) -> str | None:
        """Get auth manager name, allowing config override.

        Args:
            context: Plugin context containing config

        Returns:
            Auth manager name or None if not configured
        """
        # Check if plugin config overrides auth manager
        if hasattr(context, "config") and context.config:
            config_auth_manager = getattr(context.config, "auth_manager", None)
            if config_auth_manager:
                return str(config_auth_manager)

        # Use plugin's default auth manager name
        return self.auth_manager_name

    def create_context(self, service_container: "ServiceContainer") -> PluginContext:
        """Create context with provider-specific components.

        This method provides a hook for subclasses to customize context creation.
        The default implementation just returns the base context.

        Args:
            core_services: Core services container

        Returns:
            Plugin context
        """
        context = super().create_context(service_container)
        config = context.get("config", None)
        if isinstance(config, ProviderConfig) and config.model_mappings:
            context.model_mapper = ModelMapper(config.model_mappings)
        return context


class PluginRegistry:
    """Registry for managing plugin factories and runtime instances."""

    def __init__(self) -> None:
        """Initialize plugin registry."""
        self.factories: dict[str, PluginFactory] = {}
        self.runtimes: dict[str, Any] = {}
        self.initialization_order: list[str] = []

        # Service management
        self._services: dict[str, Any] = {}
        self._service_providers: dict[str, str] = {}  # service_name -> plugin_name

    def register_service(
        self, service_name: str, service_instance: Any, provider_plugin: str
    ) -> None:
        """Register a service provided by a plugin.

        Args:
            service_name: Name of the service
            service_instance: Service instance
            provider_plugin: Name of the plugin providing the service
        """
        if service_name in self._services:
            logger.warning(
                "service_already_registered",
                service=service_name,
                existing_provider=self._service_providers[service_name],
                new_provider=provider_plugin,
            )
        self._services[service_name] = service_instance
        self._service_providers[service_name] = provider_plugin

    def get_service(
        self, service_name: str, service_type: type[T] | None = None
    ) -> T | None:
        """Get a service by name with optional type checking.

        Args:
            service_name: Name of the service
            service_type: Optional expected service type

        Returns:
            Service instance or None if not found
        """
        service = self._services.get(service_name)
        if service and service_type and not isinstance(service, service_type):
            logger.warning(
                "service_type_mismatch",
                service=service_name,
                expected_type=service_type,
                actual_type=type(service),
            )
            return None
        return service

    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered.

        Args:
            service_name: Name of the service

        Returns:
            True if service is registered
        """
        return service_name in self._services

    def get_required_services(self, plugin_name: str) -> tuple[list[str], list[str]]:
        """Get required and optional services for a plugin.

        Args:
            plugin_name: Name of the plugin

        Returns:
            Tuple of (required_services, optional_services)
        """
        manifest = self.factories[plugin_name].get_manifest()
        return manifest.requires, manifest.optional_requires

    def register_factory(self, factory: PluginFactory) -> None:
        """Register a plugin factory.

        Args:
            factory: Plugin factory to register
        """
        manifest = factory.get_manifest()

        if manifest.name in self.factories:
            raise ValueError(f"Plugin {manifest.name} already registered")

        self.factories[manifest.name] = factory

    def get_factory(self, name: str) -> PluginFactory | None:
        """Get a plugin factory by name.

        Args:
            name: Plugin name

        Returns:
            Plugin factory or None
        """
        return self.factories.get(name)

    def get_all_manifests(self) -> dict[str, PluginManifest]:
        """Get all registered plugin manifests.

        Returns:
            Dictionary mapping plugin names to manifests
        """
        return {
            name: factory.get_manifest() for name, factory in self.factories.items()
        }

    def resolve_dependencies(self, settings: "Settings") -> list[str]:
        """Resolve plugin dependencies and return initialization order.

        Skips plugins with missing hard dependencies or required services
        instead of failing the entire plugin system. Logs skipped plugins
        and continues with the rest.

        Args:
            settings: Settings instance

        Returns:
            List of plugin names in initialization order
        """
        manifests = self.get_all_manifests()

        # Start with all plugins available
        available = set(manifests.keys())
        skipped: dict[str, str] = {}

        # Validate format adapter dependencies (latest behavior)
        missing_format_adapters = self._validate_format_adapter_requirements()
        if missing_format_adapters:
            for plugin_name, missing in missing_format_adapters.items():
                logger.error(
                    "plugin_missing_format_adapters",
                    plugin=plugin_name,
                    missing_adapters=missing,
                    category="format",
                )
                # Remove plugins with missing format adapter requirements
                available.discard(plugin_name)
                skipped[plugin_name] = f"missing format adapters: {missing}"

        # Iteratively prune plugins with unsatisfied dependencies or services
        while True:
            removed_this_pass: set[str] = set()

            # Compute services provided by currently available plugins
            available_services = {
                service for name in available for service in manifests[name].provides
            }

            for name in sorted(available):
                manifest = manifests[name]

                # Check plugin dependencies
                missing_plugins = [
                    dep for dep in manifest.dependencies if dep not in available
                ]
                if missing_plugins:
                    removed_this_pass.add(name)
                    skipped[name] = f"missing plugin dependencies: {missing_plugins}"
                    continue

                # Check required services
                missing_services = manifest.validate_service_dependencies(
                    available_services
                )
                if missing_services:
                    removed_this_pass.add(name)
                    skipped[name] = f"missing required services: {missing_services}"

            if not removed_this_pass:
                break

            # Remove the failing plugins and repeat until stable
            available -= removed_this_pass

        # Before sorting, ensure provider plugins load before consumers by
        # adding provider plugins to the consumer's dependency list.
        # Choose a stable provider (lexicographically first) when multiple exist.
        for name in available:
            manifest = manifests[name]
            for required_service in manifest.requires:
                provider_names = [
                    other_name
                    for other_name in available
                    if required_service in manifests[other_name].provides
                ]
                if provider_names:
                    provider_names.sort()
                    provider = provider_names[0]
                    if provider != name and provider not in manifest.dependencies:
                        manifest.dependencies.append(provider)

        # Kahn's algorithm for topological sort over remaining plugins
        # Build dependency graph restricted to available plugins
        deps: dict[str, list[str]] = {
            name: [dep for dep in manifests[name].dependencies if dep in available]
            for name in available
        }
        in_degree: dict[str, int] = {name: len(deps[name]) for name in available}
        dependents: dict[str, list[str]] = {name: [] for name in available}
        for name, dlist in deps.items():
            for dep in dlist:
                dependents[dep].append(name)

        # Initialize queue with nodes having zero in-degree
        queue = [name for name, deg in in_degree.items() if deg == 0]
        queue.sort()

        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for consumer in dependents[node]:
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    queue.append(consumer)
            queue.sort()

        # Any nodes not in order are part of cycles; skip them
        cyclic = [name for name in available if name not in order]
        if cyclic:
            for name in cyclic:
                skipped[name] = "circular dependency"
            logger.error(
                "plugin_dependency_cycle_detected",
                skipped=cyclic,
                category="plugin",
            )

        # Final initialization order excludes skipped and cyclic plugins
        self.initialization_order = order

        if skipped:
            logger.warning(
                "plugins_skipped_due_to_missing_dependencies",
                skipped=skipped,
                category="plugin",
            )

        return order

    def _register_auth_manager_with_registry(
        self,
        factory: AuthProviderPluginFactory,
        context: PluginContext,
    ) -> None:
        """Ensure auth provider plugins publish their managers via the registry."""

        service_container = context.get("service_container")
        if not service_container or not hasattr(
            service_container, "get_auth_manager_registry"
        ):
            logger.warning(
                "auth_manager_registry_unavailable",
                plugin=factory.get_manifest().name,
                auth_manager_name=factory.get_auth_manager_registry_name(),
                category="auth",
            )
            return

        registry = service_container.get_auth_manager_registry()
        manager_name = factory.get_auth_manager_registry_name()

        if not manager_name:
            return

        if registry.has(manager_name):
            registry.unregister(manager_name)

        existing_manager = context.get("token_manager")
        if existing_manager:
            registry.register_instance(manager_name, existing_manager)
            logger.debug(
                "auth_manager_instance_registered",
                plugin=factory.get_manifest().name,
                auth_manager_name=manager_name,
                category="auth",
            )
            return

        auth_provider = context.get("auth_provider")
        if auth_provider and hasattr(auth_provider, "create_token_manager"):

            async def manager_factory() -> Any:
                try:
                    candidate = auth_provider.create_token_manager()
                    return (
                        await candidate if inspect.isawaitable(candidate) else candidate
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "auth_manager_factory_failed",
                        plugin=factory.get_manifest().name,
                        auth_manager_name=manager_name,
                        error=str(exc),
                        exc_info=exc,
                        category="auth",
                    )
                    raise

            registry.register_factory(manager_name, manager_factory)
            logger.debug(
                "auth_manager_factory_registered",
                plugin=factory.get_manifest().name,
                auth_manager_name=manager_name,
                source="provider",
                category="auth",
            )
            return

        manager_class = getattr(factory, "auth_manager_class", None)
        if manager_class:
            registry.register_class(manager_name, manager_class)
            logger.debug(
                "auth_manager_class_registered_from_factory",
                plugin=factory.get_manifest().name,
                auth_manager_name=manager_name,
                class_name=manager_class.__name__,
                category="auth",
            )
            return

        logger.warning(
            "auth_manager_registration_missing",
            plugin=factory.get_manifest().name,
            auth_manager_name=manager_name,
            category="auth",
        )

    def _validate_format_adapter_requirements(self) -> dict[str, list[tuple[str, str]]]:
        """Self-contained helper for format adapter requirement validation.

        This method is called during dependency resolution when core_services
        is not yet available. In practice, format adapter validation happens
        later in the initialization process when the format registry is available.
        """
        # During dependency resolution phase, format registry may not be available yet
        # Return empty dict to allow dependency resolution to continue
        # Actual format adapter validation happens during initialize_all()
        logger.debug(
            "format_adapter_requirements_validation_deferred",
            message="Format adapter validation will happen during plugin initialization",
            category="format",
        )
        return {}

    async def create_runtime(
        self, name: str, service_container: "ServiceContainer"
    ) -> Any:
        """Create and initialize a plugin runtime.

        Args:
            name: Plugin name
            service_container: Service container with all available services

        Returns:
            Initialized plugin runtime

        Raises:
            ValueError: If plugin not found
        """
        factory = self.get_factory(name)
        if not factory:
            raise ValueError(f"Plugin {name} not found")

        # Check if already created
        if name in self.runtimes:
            return self.runtimes[name]

        # Create runtime instance
        runtime = factory.create_runtime()

        # Create context
        context = factory.create_context(service_container)

        # For auth provider plugins, create auth components first so registries are ready
        if isinstance(factory, AuthProviderPluginFactory):
            context.auth_provider = factory.create_auth_provider(context)
            context.token_manager = factory.create_token_manager()
            context.storage = factory.create_storage()
            self._register_auth_manager_with_registry(factory, context)

        # For provider plugins, create additional components (may depend on auth registry)
        if isinstance(factory, ProviderPluginFactory):
            # Create credentials manager and detection service first as adapter may depend on them
            context.detection_service = factory.create_detection_service(context)
            context.credentials_manager = await factory.create_credentials_manager(
                context
            )
            context.adapter = await factory.create_adapter(context)

        # Initialize runtime
        await runtime.initialize(context)

        # Store runtime
        self.runtimes[name] = runtime

        return runtime

    async def initialize_all(self, service_container: "ServiceContainer") -> None:
        """Initialize all registered plugins with format adapter support.

        Args:
            service_container: Service container with all available services
        """

        # Resolve dependencies and get initialization order
        settings = service_container.settings
        order = self.resolve_dependencies(settings)

        # Consolidated discovery summary at INFO
        logger.info(
            "plugins_discovered", count=len(order), names=order, category="plugin"
        )

        # Register format adapters from manifests in first pass (latest behavior)
        format_registry = service_container.get_format_registry()
        manifests = self.get_all_manifests()
        for name, manifest in manifests.items():
            if manifest.format_adapters:
                await format_registry.register_from_manifest(manifest, name)
                logger.debug(
                    "plugin_format_adapters_registered_from_manifest",
                    plugin=name,
                    adapter_count=len(manifest.format_adapters),
                    category="format",
                )

        # Auth managers are registered when auth provider contexts are constructed

        initialized: list[str] = []
        for name in order:
            try:
                await self.create_runtime(name, service_container)
                initialized.append(name)
            except Exception as e:
                logger.warning(
                    "plugin_initialization_failed",
                    plugin=name,
                    error=str(e),
                    exc_info=e,
                    category="plugin",
                )
                # Continue with other plugins

        # Registry entries are available immediately; log consolidated summary
        skipped = [n for n in order if n not in initialized]
        logger.info(
            "plugins_initialized",
            count=len(initialized),
            names=initialized,
            skipped=skipped if skipped else [],
            category="plugin",
        )

        # Emit a single hooks summary at the end
        try:
            hook_registry = service_container.get_hook_registry()
            totals: dict[str, int] = {}
            for event_name, hooks in hook_registry.list().items():
                totals[event_name] = len(hooks)
            logger.info(
                "hooks_registered",
                total_events=len(totals),
                by_event_counts=totals,
            )
        except Exception:
            pass

    async def shutdown_all(self) -> None:
        """Shutdown all plugin runtimes in reverse initialization order."""
        # Shutdown in reverse order
        for name in reversed(self.initialization_order):
            if name in self.runtimes:
                runtime = self.runtimes[name]
                try:
                    await runtime.shutdown()
                except Exception as e:
                    logger.error(
                        "plugin_shutdown_failed",
                        plugin=name,
                        error=str(e),
                        exc_info=e,
                        category="plugin",
                    )

        # Clear runtimes
        self.runtimes.clear()

    def get_runtime(self, name: str) -> Any | None:
        """Get a plugin runtime by name.

        Args:
            name: Plugin name

        Returns:
            Plugin runtime or None
        """
        return self.runtimes.get(name)

    def list_plugins(self) -> list[str]:
        """List all registered plugin names.

        Returns:
            List of plugin names
        """
        return list(self.factories.keys())

    def list_provider_plugins(self) -> list[str]:
        """List all registered provider plugin names.

        Returns:
            List of provider plugin names
        """
        return [
            name
            for name, factory in self.factories.items()
            if factory.get_manifest().is_provider
        ]
