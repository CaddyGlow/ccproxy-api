"""Plugin factory protocol for bridging declaration and runtime.

This module provides the factory layer that creates runtime instances
from plugin manifests and manages the plugin lifecycle.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

import structlog

from .declaration import PluginContext, PluginManifest
from .runtime import (
    AuthProviderPluginRuntime,
    BasePluginRuntime,
    ProviderPluginRuntime,
    SystemPluginRuntime,
)


logger = structlog.get_logger(__name__)

# Type variable for service type checking
T = TypeVar("T")


class PluginFactory(ABC):
    """Abstract factory for creating plugin runtime instances.

    Each plugin must provide a factory that knows how to create
    its runtime instance from its manifest.
    """

    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Get the plugin manifest with static declarations.

        Returns:
            Plugin manifest
        """
        ...

    @abstractmethod
    def create_runtime(self) -> BasePluginRuntime:
        """Create a runtime instance for this plugin.

        Returns:
            Plugin runtime instance
        """
        ...

    @abstractmethod
    def create_context(self, core_services: Any) -> PluginContext:
        """Create the context for plugin initialization.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with required services
        """
        ...


class BasePluginFactory(PluginFactory):
    """Base implementation of plugin factory.

    This class provides common functionality for creating plugin
    runtime instances from manifests.
    """

    def __init__(
        self, manifest: PluginManifest, runtime_class: type[BasePluginRuntime]
    ):
        """Initialize factory with manifest and runtime class.

        Args:
            manifest: Plugin manifest
            runtime_class: Runtime class to instantiate
        """
        self.manifest = manifest
        self.runtime_class = runtime_class

    def get_manifest(self) -> PluginManifest:
        """Get the plugin manifest."""
        return self.manifest

    def create_runtime(self) -> BasePluginRuntime:
        """Create a runtime instance."""
        return self.runtime_class(self.manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        """Create base context for plugin initialization.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with base services
        """
        context: PluginContext = {
            "settings": core_services.settings,
            "http_pool_manager": core_services.http_pool_manager,
            "logger": core_services.logger.bind(plugin=self.manifest.name),
        }

        # Add explicit dependency injection services
        if hasattr(core_services, "request_tracer"):
            context["request_tracer"] = core_services.request_tracer
        if hasattr(core_services, "streaming_handler"):
            context["streaming_handler"] = core_services.streaming_handler
        if hasattr(core_services, "metrics"):
            context["metrics"] = core_services.metrics

        # Add CLI detection service if available
        if hasattr(core_services, "cli_detection_service"):
            context["cli_detection_service"] = core_services.cli_detection_service

        # Add scheduler if available
        if hasattr(core_services, "scheduler"):
            context["scheduler"] = core_services.scheduler

        # Add plugin registry (SINGLE SOURCE for all plugin/service access)
        if hasattr(core_services, "plugin_registry"):
            context["plugin_registry"] = core_services.plugin_registry

        # Add OAuth registry for auth providers if available (avoid globals)
        if (
            hasattr(core_services, "oauth_registry")
            and core_services.oauth_registry is not None
        ):
            context["oauth_registry"] = core_services.oauth_registry

        # Add hook registry and manager if available
        if hasattr(core_services, "hook_registry"):
            context["hook_registry"] = core_services.hook_registry
        if hasattr(core_services, "hook_manager"):
            context["hook_manager"] = core_services.hook_manager
        if hasattr(core_services, "app"):
            context["app"] = core_services.app

        # Add plugin-specific config if available
        if hasattr(core_services, "get_plugin_config"):
            plugin_config = core_services.get_plugin_config(self.manifest.name)
            if plugin_config and self.manifest.config_class:
                # Validate config with plugin's config class
                validated_config = self.manifest.config_class.model_validate(
                    plugin_config
                )
                context["config"] = validated_config

        return context


class SystemPluginFactory(BasePluginFactory):
    """Factory for system plugins."""

    def __init__(self, manifest: PluginManifest):
        """Initialize system plugin factory.

        Args:
            manifest: Plugin manifest
        """
        super().__init__(manifest, SystemPluginRuntime)

        # Validate this is a system plugin
        if manifest.is_provider:
            raise ValueError(
                f"Plugin {manifest.name} is marked as provider but using SystemPluginFactory"
            )


class ProviderPluginFactory(BasePluginFactory):
    """Factory for provider plugins.

    Provider plugins require additional components like adapters
    and detection services that must be created during initialization.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize provider plugin factory.

        Args:
            manifest: Plugin manifest
        """
        super().__init__(manifest, ProviderPluginRuntime)

        # Validate this is a provider plugin
        if not manifest.is_provider:
            raise ValueError(
                f"Plugin {manifest.name} is not marked as provider but using ProviderPluginFactory"
            )

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with provider-specific components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with provider components
        """
        # Start with base context
        context = super().create_context(core_services)

        # Provider plugins need to create their own adapter and detection service
        # This is typically done in the specific plugin factory implementation
        # Here we just ensure the structure is correct

        return context

    @abstractmethod
    async def create_adapter(self, context: PluginContext) -> Any:
        """Create the adapter for this provider.

        Args:
            context: Plugin context

        Returns:
            Provider adapter instance
        """
        ...

    @abstractmethod
    def create_detection_service(self, context: PluginContext) -> Any:
        """Create the detection service for this provider.

        Args:
            context: Plugin context

        Returns:
            Detection service instance or None
        """
        ...

    @abstractmethod
    def create_credentials_manager(self, context: PluginContext) -> Any:
        """Create the credentials manager for this provider.

        Args:
            context: Plugin context

        Returns:
            Credentials manager instance or None
        """
        ...


class AuthProviderPluginFactory(BasePluginFactory):
    """Factory for authentication provider plugins.

    Auth provider plugins provide OAuth authentication flows and token management
    without directly proxying requests to API providers.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize auth provider plugin factory.

        Args:
            manifest: Plugin manifest
        """
        super().__init__(manifest, AuthProviderPluginRuntime)

        # Validate this is marked as a provider plugin (auth providers are a type of provider)
        if not manifest.is_provider:
            raise ValueError(
                f"Plugin {manifest.name} must be marked as provider for AuthProviderPluginFactory"
            )

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with auth provider-specific components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with auth provider components
        """
        # Start with base context
        context = super().create_context(core_services)

        # Auth provider plugins need to create their auth components
        # This is typically done in the specific plugin factory implementation

        return context

    @abstractmethod
    def create_auth_provider(self, context: PluginContext | None = None) -> Any:
        """Create the OAuth provider for this auth plugin.

        Args:
            context: Optional plugin context for initialization

        Returns:
            OAuth provider instance implementing OAuthProviderProtocol
        """
        ...

    def create_token_manager(self) -> Any | None:
        """Create the token manager for this auth plugin.

        Returns:
            Token manager instance or None if not needed
        """
        return None

    def create_storage(self) -> Any | None:
        """Create the storage implementation for this auth plugin.

        Returns:
            Storage instance or None if using default
        """
        return None


def factory_type_name(factory: PluginFactory) -> str:
    """Return a stable type name for a plugin factory.

    Returns one of: "auth_provider", "provider", "system", or "plugin" (fallback).
    """
    try:
        if isinstance(factory, AuthProviderPluginFactory):
            return "auth_provider"
        if isinstance(factory, ProviderPluginFactory):
            return "provider"
        if isinstance(factory, SystemPluginFactory):
            return "system"
    except Exception:
        pass
    return "plugin"


class PluginRegistry:
    """Registry for managing plugin factories and runtime instances."""

    def __init__(self) -> None:
        """Initialize plugin registry."""
        self.factories: dict[str, PluginFactory] = {}
        self.runtimes: dict[str, BasePluginRuntime] = {}
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
        # logger.debug(
        #     "service_registered", service=service_name, provider=provider_plugin
        # )

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
        # logger.debug(
        #     "plugin_factory_registered",
        #     plugin=manifest.name,
        #     version=manifest.version,
        #     is_provider=manifest.is_provider,
        #     category="plugin",
        # )

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

    def resolve_dependencies(self) -> list[str]:
        """Resolve plugin dependencies and return initialization order.

        Skips plugins with missing hard dependencies or required services
        instead of failing the entire plugin system. Logs skipped plugins
        and continues with the rest.

        Returns:
            List of plugin names in initialization order
        """
        manifests = self.get_all_manifests()

        # Start with all plugins available
        available = set(manifests.keys())
        skipped: dict[str, str] = {}

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

    async def create_runtime(self, name: str, core_services: Any) -> BasePluginRuntime:
        """Create and initialize a plugin runtime.

        Args:
            name: Plugin name
            core_services: Core services container

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
        context = factory.create_context(core_services)

        # For provider plugins, create additional components
        if isinstance(factory, ProviderPluginFactory):
            # Create credentials manager and detection service first as adapter may depend on them
            context["detection_service"] = factory.create_detection_service(context)
            context["credentials_manager"] = factory.create_credentials_manager(context)
            context["adapter"] = await factory.create_adapter(context)
        # For auth provider plugins, create auth components
        elif isinstance(factory, AuthProviderPluginFactory):
            context["auth_provider"] = factory.create_auth_provider(context)
            context["token_manager"] = factory.create_token_manager()
            context["storage"] = factory.create_storage()

        # Initialize runtime
        await runtime.initialize(context)

        # Store runtime
        self.runtimes[name] = runtime

        return runtime

    async def initialize_all(self, core_services: Any) -> None:
        """Initialize all registered plugins in dependency order.

        Args:
            core_services: Core services container
        """
        order = self.resolve_dependencies()

        logger.info(
            "initializing_plugins", count=len(order), order=order, category="plugin"
        )

        for name in order:
            try:
                await self.create_runtime(name, core_services)
            except Exception as e:
                logger.warning(
                    "plugin_initialization_failed",
                    plugin=name,
                    error=str(e),
                    exc_info=e,
                    category="plugin",
                )
                # Continue with other plugins

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

    def get_runtime(self, name: str) -> BasePluginRuntime | None:
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
