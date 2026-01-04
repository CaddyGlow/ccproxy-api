"""Plugin runtime system for managing plugin instances.

This module defines runtime classes that manage plugin instances and lifecycle.
Factory/loader utilities remain in their respective modules to avoid import
cycles during consolidation. Import runtime classes from here, and import
factories/loaders from their modules for now.
"""

from typing import Any

from ccproxy.core.logging import TraceBoundLogger, get_logger

from .declaration import PluginContext, PluginManifest, PluginRuntimeProtocol


__all__ = [
    "BasePluginRuntime",
    "SystemPluginRuntime",
    "AuthProviderPluginRuntime",
    "ProviderPluginRuntime",
    "PluginContext",
    "PluginManifest",
    "PluginRuntimeProtocol",
]


logger: TraceBoundLogger = get_logger()


class BasePluginRuntime(PluginRuntimeProtocol):
    """Base implementation of plugin runtime.

    This class provides common functionality for all plugin runtimes.
    Specific plugin types (system, provider) can extend this base class.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime with manifest.

        Args:
            manifest: Plugin manifest with static declarations
        """
        self.manifest = manifest
        self.context: PluginContext | None = None
        self.initialized = False

    @property
    def name(self) -> str:
        """Plugin name from manifest."""
        return self.manifest.name

    @property
    def version(self) -> str:
        """Plugin version from manifest."""
        return self.manifest.version

    async def initialize(self, context: PluginContext) -> None:
        """Initialize the plugin with runtime context.

        Args:
            context: Runtime context with services and configuration
        """
        if self.initialized:
            logger.warning(
                "plugin_already_initialized", plugin=self.name, category="plugin"
            )
            return

        self.context = context

        # Allow subclasses to perform custom initialization
        await self._on_initialize()

        self.initialized = True
        logger.debug(
            "plugin_initialized",
            plugin=self.name,
            version=self.version,
            category="plugin",
        )

    async def _on_initialize(self) -> None:
        """Hook for subclasses to perform custom initialization.

        Override this method in subclasses to add custom initialization logic.
        """
        pass

    async def shutdown(self) -> None:
        """Cleanup on shutdown."""
        if not self.initialized:
            return

        # Allow subclasses to perform custom cleanup
        await self._on_shutdown()

        self.initialized = False
        logger.info("plugin_shutdown", plugin=self.name, category="plugin")

    async def _on_shutdown(self) -> None:
        """Hook for subclasses to perform custom cleanup.

        Override this method in subclasses to add custom cleanup logic.
        """
        pass

    async def validate(self) -> bool:
        """Validate plugin is ready.

        Returns:
            True if plugin is ready, False otherwise
        """
        # Basic validation - plugin is initialized
        if not self.initialized:
            return False

        # Allow subclasses to add custom validation
        return await self._on_validate()

    async def _on_validate(self) -> bool:
        """Hook for subclasses to perform custom validation.

        Override this method in subclasses to add custom validation logic.

        Returns:
            True if validation passes, False otherwise
        """
        return True

    async def health_check(self) -> dict[str, Any]:
        """Perform health check.

        Returns:
            Health check result following IETF format
        """
        try:
            # Start with basic health check
            is_healthy = await self.validate()

            # Allow subclasses to provide detailed health info
            details = await self._get_health_details()

            return {
                "status": "pass" if is_healthy else "fail",
                "componentId": self.name,
                "componentType": "provider_plugin"
                if self.manifest.is_provider
                else "system_plugin",
                "version": self.version,
                "details": details,
            }
        except Exception as e:
            logger.error(
                "plugin_health_check_failed",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )
            return {
                "status": "fail",
                "componentId": self.name,
                "componentType": "provider_plugin"
                if self.manifest.is_provider
                else "system_plugin",
                "version": self.version,
                "output": str(e),
            }

    async def _get_health_details(self) -> dict[str, Any]:
        """Hook for subclasses to provide health check details.

        Override this method in subclasses to add custom health check details.

        Returns:
            Dictionary with health check details
        """
        return {}


class SystemPluginRuntime(BasePluginRuntime):
    """Runtime for system plugins (non-provider plugins).

    System plugins provide functionality like logging, monitoring,
    permissions, etc., but don't proxy to external providers.
    """

    async def _on_initialize(self) -> None:
        """System plugin initialization."""
        logger.debug("system_plugin_initializing", plugin=self.name, category="plugin")
        # System plugins typically don't need special initialization
        # but can override this method if needed

    async def _get_health_details(self) -> dict[str, Any]:
        """System plugin health details."""
        return {"type": "system", "initialized": self.initialized}


class AuthProviderPluginRuntime(BasePluginRuntime):
    """Runtime for authentication provider plugins.

    Auth provider plugins provide OAuth authentication flows and token management
    for various API providers without directly proxying requests.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize auth provider plugin runtime.

        Args:
            manifest: Plugin manifest with static declarations
        """
        super().__init__(manifest)
        self.auth_provider: Any | None = None  # OAuthProviderProtocol
        self.token_manager: Any | None = None
        self.storage: Any | None = None

    async def _on_initialize(self) -> None:
        """Auth provider plugin initialization."""
        logger.debug(
            "auth_provider_plugin_initializing", plugin=self.name, category="plugin"
        )

        if not self.context:
            raise RuntimeError("Context not set")

        # Extract auth-specific components from context
        self.auth_provider = self.context.get("auth_provider")
        self.token_manager = self.context.get("token_manager")
        self.storage = self.context.get("storage")

        # Register OAuth provider with app-scoped registry if present
        if self.auth_provider:
            await self._register_auth_provider()

    async def _register_auth_provider(self) -> None:
        """Register OAuth provider with the app-scoped registry."""
        if not self.auth_provider:
            return

        try:
            # Register with app-scoped registry from context
            registry = None
            if self.context and "oauth_registry" in self.context:
                registry = self.context["oauth_registry"]
            if registry is None:
                logger.warning(
                    "oauth_registry_missing_in_context",
                    plugin=self.name,
                    category="plugin",
                )
                return
            registry.register(self.auth_provider)

            logger.debug(
                "oauth_provider_registered",
                plugin=self.name,
                provider=self.auth_provider.provider_name,
                category="plugin",
            )
        except Exception as e:
            logger.error(
                "oauth_provider_registration_failed",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

    async def _on_shutdown(self) -> None:
        """Auth provider plugin shutdown."""
        # Cleanup provider resources if it has a cleanup method
        if self.auth_provider and hasattr(self.auth_provider, "cleanup"):
            try:
                await self.auth_provider.cleanup()
                logger.debug(
                    "oauth_provider_cleaned_up",
                    plugin=self.name,
                    provider=self.auth_provider.provider_name,
                    category="plugin",
                )
            except Exception as e:
                logger.error(
                    "oauth_provider_cleanup_failed",
                    plugin=self.name,
                    error=str(e),
                    exc_info=e,
                    category="plugin",
                )

        # Unregister OAuth provider if present
        if self.auth_provider:
            await self._unregister_auth_provider()

    async def _unregister_auth_provider(self) -> None:
        """Unregister OAuth provider from the app-scoped registry."""
        if not self.auth_provider:
            return

        try:
            # Unregister from app-scoped registry available in context
            registry = None
            if self.context and "oauth_registry" in self.context:
                registry = self.context["oauth_registry"]
            if registry is None:
                logger.warning(
                    "oauth_registry_missing_in_context_on_shutdown",
                    plugin=self.name,
                    category="plugin",
                )
                return
            registry.unregister(self.auth_provider.provider_name)

            logger.debug(
                "oauth_provider_unregistered",
                plugin=self.name,
                provider=self.auth_provider.provider_name,
                category="plugin",
            )
        except Exception as e:
            logger.error(
                "oauth_provider_unregistration_failed",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

    async def _get_health_details(self) -> dict[str, Any]:
        """Auth provider plugin health details."""
        details = {
            "type": "auth_provider",
            "initialized": self.initialized,
        }

        if self.auth_provider:
            # Check if provider is registered
            try:
                registry = None
                if self.context and "oauth_registry" in self.context:
                    registry = self.context["oauth_registry"]
                is_registered = (
                    registry.has(self.auth_provider.provider_name)
                    if registry is not None
                    else False
                )
                details.update(
                    {
                        "oauth_provider_registered": is_registered,
                        "oauth_provider_name": self.auth_provider.provider_name,
                    }
                )
            except Exception:
                pass

        return details


class ProviderPluginRuntime(BasePluginRuntime):
    """Runtime for provider plugins.

    Provider plugins proxy requests to external API providers and
    require additional components like adapters and detection services.
    """

    def __init__(self, manifest: PluginManifest):
        """Initialize provider plugin runtime.

        Args:
            manifest: Plugin manifest with static declarations
        """
        super().__init__(manifest)
        self.adapter: Any | None = None  # BaseAdapter
        self.detection_service: Any | None = None
        self.credentials_manager: Any | None = None

    async def _on_initialize(self) -> None:
        """Provider plugin initialization."""
        logger.debug(
            "provider_plugin_initializing", plugin=self.name, category="plugin"
        )

        if not self.context:
            raise RuntimeError("Context not set")

        # Extract provider-specific components from context
        self.adapter = self.context.get("adapter")
        self.detection_service = self.context.get("detection_service")
        self.credentials_manager = self.context.get("credentials_manager")

        # Initialize detection service if present
        if self.detection_service and hasattr(
            self.detection_service, "initialize_detection"
        ):
            await self.detection_service.initialize_detection()
            logger.debug(
                "detection_service_initialized", plugin=self.name, category="plugin"
            )

        # Register OAuth provider if factory is provided
        if self.manifest.oauth_provider_factory:
            await self._register_oauth_provider()

    async def _register_oauth_provider(self) -> None:
        """Register OAuth provider with the app-scoped registry."""
        if not self.manifest.oauth_provider_factory:
            return

        try:
            # Create OAuth provider instance
            oauth_provider = self.manifest.oauth_provider_factory()

            # Use oauth_registry from context (injected via core services)
            registry = None
            if self.context and "oauth_registry" in self.context:
                registry = self.context["oauth_registry"]

            if registry is None:
                logger.warning(
                    "oauth_registry_missing_in_context",
                    plugin=self.name,
                    category="plugin",
                )
                return

            registry.register(oauth_provider)

            logger.trace(
                "oauth_provider_registered",
                plugin=self.name,
                provider=oauth_provider.provider_name,
                category="plugin",
            )
        except Exception as e:
            logger.error(
                "oauth_provider_registration_failed",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

    async def _unregister_oauth_provider(self) -> None:
        """Unregister OAuth provider from the app-scoped registry."""
        if not self.manifest.oauth_provider_factory:
            return

        try:
            # Determine provider name
            oauth_provider = self.manifest.oauth_provider_factory()
            provider_name = oauth_provider.provider_name

            # Use oauth_registry from context (injected via core services)
            registry = None
            if self.context and "oauth_registry" in self.context:
                registry = self.context["oauth_registry"]

            if registry is None:
                logger.warning(
                    "oauth_registry_missing_in_context_on_shutdown",
                    plugin=self.name,
                    category="plugin",
                )
                return

            registry.unregister(provider_name)

            logger.trace(
                "oauth_provider_unregistered",
                plugin=self.name,
                provider=provider_name,
                category="plugin",
            )
        except Exception as e:
            logger.error(
                "oauth_provider_unregistration_failed",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

    async def _on_shutdown(self) -> None:
        """Provider plugin cleanup."""
        # Unregister OAuth provider if registered
        await self._unregister_oauth_provider()

        # Cleanup adapter if present
        if self.adapter and hasattr(self.adapter, "cleanup"):
            await self.adapter.cleanup()
            logger.debug("adapter_cleaned_up", plugin=self.name, category="plugin")

    async def _on_validate(self) -> bool:
        """Provider plugin validation."""
        # Check that required components are present
        if self.manifest.is_provider and not self.adapter:
            logger.warning(
                "provider_plugin_missing_adapter", plugin=self.name, category="plugin"
            )
            return False
        return True

    async def _get_health_details(self) -> dict[str, Any]:
        """Provider plugin health details."""
        details: dict[str, Any] = {
            "type": "provider",
            "initialized": self.initialized,
            "has_adapter": self.adapter is not None,
            "has_detection": self.detection_service is not None,
            "has_credentials": self.credentials_manager is not None,
        }

        # Add detection service info if available
        if self.detection_service:
            if hasattr(self.detection_service, "get_version"):
                details["cli_version"] = self.detection_service.get_version()
            if hasattr(self.detection_service, "get_cli_path"):
                details["cli_path"] = self.detection_service.get_cli_path()

        return details
