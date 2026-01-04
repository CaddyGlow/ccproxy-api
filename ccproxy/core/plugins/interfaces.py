"""Abstract interfaces for the plugin system.

This module contains all abstract base classes and protocols to avoid
circular dependencies between factory and runtime modules.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import structlog
from fastapi import FastAPI

from ccproxy.auth import TokenStorage
from ccproxy.auth.managers import BaseTokenManager, TokenSnapshot
from ccproxy.auth.oauth import OAuthProviderProtocol
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts


if TYPE_CHECKING:
    from ccproxy.services.container import ServiceContainer
    from ccproxy.services.interfaces import StreamingMetrics

from .declaration import PluginContext, PluginManifest


# Type variable for service type checking
T = TypeVar("T")

logger = structlog.get_logger(__name__)

# --- Adapter protocol helpers -------------------------------------------------


class DetectionServiceProtocol(Protocol):
    """Common capabilities shared by detection services."""

    def get_detected_headers(self) -> DetectedHeaders: ...

    def get_detected_prompts(self) -> DetectedPrompts: ...

    def get_cached_data(self) -> object: ...

    def get_system_prompt(self, mode: str | None = None) -> dict[str, object]: ...

    def get_ignored_headers(self) -> list[str]: ...

    def get_redacted_headers(self) -> list[str]: ...


class TokenManagerProtocol(Protocol):
    """Minimal async token manager contract used by adapters."""

    async def get_access_token(self) -> str | None: ...

    async def get_access_token_with_refresh(self) -> str | None: ...

    async def load_credentials(self) -> object: ...

    async def get_token_snapshot(self) -> TokenSnapshot | None: ...

    def should_refresh(
        self, credentials: object, grace_seconds: float | None = None
    ) -> bool: ...


class ProfiledTokenManagerProtocol(TokenManagerProtocol, Protocol):
    """Token manager that can return a lightweight profile snapshot."""

    async def get_profile_quick(self) -> object | None: ...


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
    def create_context(self, core_services: "ServiceContainer") -> PluginContext:
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

    def create_context(self, service_container: "ServiceContainer") -> PluginContext:
        """Create base context for plugin initialization.

        Args:
            service_container: Service container with all available services

        Returns:
            Plugin context with base services
        """
        context = PluginContext()

        # Set core services
        context.settings = service_container.settings
        context.http_pool_manager = service_container.get_pool_manager()
        context.logger = structlog.get_logger().bind(plugin=self.manifest.name)

        # Add explicit dependency injection services
        context.request_tracer = service_container.get_request_tracer()
        context.streaming_handler = cast(
            "StreamingMetrics", service_container.get_streaming_handler()
        )
        context.metrics = None  # Will be set by plugins if needed

        # Add CLI detection service
        context.cli_detection_service = service_container.get_cli_detection_service()

        # Add scheduler - not available in ServiceContainer, get from app state
        context.scheduler = None  # Will be set from app.state if needed

        # Add plugin registry - not directly in ServiceContainer, get from app state
        context.plugin_registry = None  # Will be set from app.state

        # Add OAuth registry for auth providers
        context.oauth_registry = service_container.get_oauth_registry()

        # Add hook registry and manager
        context.hook_registry = service_container.get_hook_registry()

        # Provide runtime helpers when available in the container
        try:
            from ccproxy.core.plugins.hooks.manager import HookManager

            context.hook_manager = service_container.get_service(HookManager)
        except (ValueError, ImportError):
            context.hook_manager = None

        try:
            context.app = service_container.get_service(FastAPI)
        except ValueError:
            context.app = None

        # Add service container directly
        context.service_container = service_container

        # Add plugin-specific config if available
        # ServiceContainer doesn't have get_plugin_config, so we'll get it from settings directly
        if self.manifest.config_class:
            plugin_config = service_container.settings.plugins.get(self.manifest.name)

            try:
                if plugin_config is None:
                    # No explicit config provided; instantiate defaults
                    context.config = self.manifest.config_class()
                else:
                    # Validate (even if empty dict) to honor model defaults
                    validated_config = self.manifest.config_class.model_validate(
                        plugin_config
                    )
                    context.config = validated_config
            except Exception as exc:  # pragma: no cover - defensive safety
                logger.warning(
                    "plugin_config_initialization_failed",
                    plugin=self.manifest.name,
                    error=str(exc),
                )
                raise

        # Add format registry
        context.format_registry = service_container.get_format_registry()

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

    def create_context(self, service_container: "ServiceContainer") -> PluginContext:
        """Create context with provider-specific components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with provider components
        """
        # Start with base context
        context = super().create_context(service_container)

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
    def create_detection_service(
        self, context: PluginContext
    ) -> DetectionServiceProtocol | None:
        """Create the detection service for this provider.

        Args:
            context: Plugin context

        Returns:
            Detection service instance or None
        """
        ...

    @abstractmethod
    async def create_credentials_manager(
        self, context: PluginContext
    ) -> BaseTokenManager[Any] | None:
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

    auth_manager_class: type[Any] | None = None

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

    def create_context(self, service_container: "ServiceContainer") -> PluginContext:
        """Create context with auth provider-specific components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with auth provider components
        """
        # Start with base context
        context = super().create_context(service_container)

        # Auth provider plugins need to create their auth components
        # This is typically done in the specific plugin factory implementation

        return context

    def get_auth_manager_registry_name(self) -> str:
        """Return registry key used for this auth manager."""

        name = getattr(self, "auth_manager_name", None)
        return name or self.manifest.name

    @abstractmethod
    def create_auth_provider(
        self, context: PluginContext | None = None
    ) -> OAuthProviderProtocol:
        """Create the OAuth provider for this auth plugin.

        Args:
            context: Optional plugin context for initialization

        Returns:
            OAuth provider instance implementing  OAuthProviderProtocol
        """
        ...

    def create_token_manager(self) -> BaseTokenManager[Any] | None:
        """Create the token manager for this auth plugin.

        Returns:
            Token manager instance or None if not needed
        """
        return None

    def create_storage(self) -> TokenStorage[Any] | None:
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
