"""Plugin factory protocol for bridging declaration and runtime.

This module provides the factory layer that creates runtime instances
from plugin manifests and manages the plugin lifecycle.
"""

from abc import ABC, abstractmethod
from typing import Any

import structlog

from .declaration import PluginContext, PluginManifest
from .runtime import BasePluginRuntime, ProviderPluginRuntime, SystemPluginRuntime


logger = structlog.get_logger(__name__)


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
            "http_client": core_services.http_client,
            "logger": core_services.logger.bind(plugin=self.manifest.name),
            "proxy_service": core_services.proxy_service,
        }

        # Add CLI detection service if available
        if hasattr(core_services, "cli_detection_service"):
            context["cli_detection_service"] = core_services.cli_detection_service

        # Add scheduler if available
        if hasattr(core_services, "scheduler"):
            context["scheduler"] = core_services.scheduler

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
    def create_adapter(self, context: PluginContext) -> Any:
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


class PluginRegistry:
    """Registry for managing plugin factories and runtime instances."""

    def __init__(self) -> None:
        """Initialize plugin registry."""
        self.factories: dict[str, PluginFactory] = {}
        self.runtimes: dict[str, BasePluginRuntime] = {}
        self.initialization_order: list[str] = []

    def register_factory(self, factory: PluginFactory) -> None:
        """Register a plugin factory.

        Args:
            factory: Plugin factory to register
        """
        manifest = factory.get_manifest()

        if manifest.name in self.factories:
            raise ValueError(f"Plugin {manifest.name} already registered")

        self.factories[manifest.name] = factory
        logger.debug(
            "plugin_factory_registered",
            plugin=manifest.name,
            version=manifest.version,
            is_provider=manifest.is_provider,
            category="plugin",
        )

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

        Returns:
            List of plugin names in initialization order

        Raises:
            ValueError: If circular dependencies detected or missing dependencies
        """
        manifests = self.get_all_manifests()
        available = set(manifests.keys())

        # Check for missing dependencies
        for name, manifest in manifests.items():
            missing = manifest.validate_dependencies(available)
            if missing:
                raise ValueError(f"Plugin {name} has missing dependencies: {missing}")

        # Topological sort for dependency resolution
        visited = set()
        temp_mark = set()
        order = []

        def visit(name: str) -> None:
            if name in temp_mark:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return

            temp_mark.add(name)
            manifest = manifests[name]

            # Visit dependencies first
            for dep in manifest.dependencies:
                if dep in manifests:  # Only visit if dependency is registered
                    visit(dep)

            temp_mark.remove(name)
            visited.add(name)
            order.append(name)

        # Visit all plugins
        for name in manifests:
            if name not in visited:
                visit(name)

        self.initialization_order = order
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
            context["adapter"] = factory.create_adapter(context)

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
                logger.error(
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
