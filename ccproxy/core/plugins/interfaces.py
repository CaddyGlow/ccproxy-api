"""Abstract interfaces for the plugin system.

This module contains all abstract base classes and protocols to avoid
circular dependencies between factory and runtime modules.
"""

import contextlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    pass

from .declaration import PluginContext, PluginManifest


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
    def create_runtime(self) -> Any:
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

    def __init__(self, manifest: PluginManifest, runtime_class: type[Any]):
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

    def create_runtime(self) -> Any:
        """Create a runtime instance."""
        return self.runtime_class(self.manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        """Create base context for plugin initialization.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with base services
        """
        context = PluginContext()

        # Set core services
        context.settings = core_services.settings
        context.http_pool_manager = core_services.http_pool_manager
        context.logger = core_services.logger.bind(plugin=self.manifest.name)

        # Add explicit dependency injection services
        if hasattr(core_services, "request_tracer"):
            context.request_tracer = core_services.request_tracer
        if hasattr(core_services, "streaming_handler"):
            context.streaming_handler = core_services.streaming_handler
        if hasattr(core_services, "metrics"):
            context.metrics = core_services.metrics

        # Add CLI detection service if available
        if hasattr(core_services, "cli_detection_service"):
            context.cli_detection_service = core_services.cli_detection_service

        # Add scheduler if available
        if hasattr(core_services, "scheduler"):
            context.scheduler = core_services.scheduler

        # Add plugin registry (SINGLE SOURCE for all plugin/service access)
        if hasattr(core_services, "plugin_registry"):
            context.plugin_registry = core_services.plugin_registry

        # Add OAuth registry for auth providers if available (avoid globals)
        if (
            hasattr(core_services, "oauth_registry")
            and core_services.oauth_registry is not None
        ):
            context.oauth_registry = core_services.oauth_registry

        # Add hook registry and manager if available
        if hasattr(core_services, "hook_registry"):
            context.hook_registry = core_services.hook_registry
        if hasattr(core_services, "hook_manager"):
            context.hook_manager = core_services.hook_manager
        if hasattr(core_services, "app"):
            context.app = core_services.app

        # Add service container if available
        if hasattr(core_services, "_container"):
            context.service_container = core_services._container

        # Add plugin-specific config if available
        if hasattr(core_services, "get_plugin_config"):
            plugin_config = core_services.get_plugin_config(self.manifest.name)
            if plugin_config and self.manifest.config_class:
                # Validate config with plugin's config class
                validated_config = self.manifest.config_class.model_validate(
                    plugin_config
                )
                context.config = validated_config

        if hasattr(core_services, "get_format_registry"):
            with contextlib.suppress(Exception):
                context.format_registry = core_services.get_format_registry()

        return context


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
        # Local import to avoid circular dependency at module load time
        from .runtime import ProviderPluginRuntime

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


class SystemPluginFactory(BasePluginFactory):
    """Factory for system plugins."""

    def __init__(self, manifest: PluginManifest):
        """Initialize system plugin factory.

        Args:
            manifest: Plugin manifest
        """
        # Local import to avoid circular dependency at module load time
        from .runtime import SystemPluginRuntime

        super().__init__(manifest, SystemPluginRuntime)

        # Validate this is a system plugin
        if manifest.is_provider:
            raise ValueError(
                f"Plugin {manifest.name} is marked as provider but using SystemPluginFactory"
            )


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
        # Local import to avoid circular dependency at module load time
        from .runtime import AuthProviderPluginRuntime

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
