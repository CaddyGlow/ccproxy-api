"""Plugin runtime system for managing plugin instances.

This module provides the runtime layer of the plugin system, managing
plugin instances and their lifecycle after the application has started.
"""

from typing import Any

from ccproxy.core.logging import TraceBoundLogger, get_logger

from .declaration import PluginContext, PluginManifest, PluginRuntimeProtocol


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

    async def get_profile_info(self) -> dict[str, Any] | None:
        """Get provider profile information.

        Default implementation returns None.
        Provider plugins should override this method.

        Returns:
            Profile information or None
        """
        return None

    async def get_auth_summary(self) -> dict[str, Any]:
        """Get authentication summary.

        Default implementation returns basic status.
        Provider plugins should override this method.

        Returns:
            Authentication summary
        """
        return {"auth": "not_applicable"}


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
        """Register OAuth provider with the global registry."""
        if not self.manifest.oauth_provider_factory:
            return

        try:
            # Import here to avoid circular dependency
            from ccproxy.auth.oauth.registry import get_oauth_registry

            # Create OAuth provider instance
            oauth_provider = self.manifest.oauth_provider_factory()

            # Register with global registry
            registry = get_oauth_registry()
            registry.register_provider(oauth_provider)

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
        """Unregister OAuth provider from the global registry."""
        if not self.manifest.oauth_provider_factory:
            return

        try:
            # Import here to avoid circular dependency
            from ccproxy.auth.oauth.registry import get_oauth_registry

            # Get provider name - we need to create a temporary instance
            # to get the name, or store it during registration
            oauth_provider = self.manifest.oauth_provider_factory()
            provider_name = oauth_provider.provider_name

            # Unregister from global registry
            registry = get_oauth_registry()
            registry.unregister_provider(provider_name)

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

    async def get_profile_info(self) -> dict[str, Any] | None:
        """Get provider profile information.

        Returns:
            Profile information from credentials manager
        """
        if not self.credentials_manager:
            return None

        try:
            # Attempt to get profile from credentials manager
            if hasattr(self.credentials_manager, "get_account_profile"):
                profile = await self.credentials_manager.get_account_profile()
                if profile:
                    return self._format_profile(profile)

            # Try to fetch fresh profile
            if hasattr(self.credentials_manager, "fetch_user_profile"):
                profile = await self.credentials_manager.fetch_user_profile()
                if profile:
                    return self._format_profile(profile)

        except Exception as e:
            logger.debug(
                "profile_fetch_error",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

        return None

    def _format_profile(self, profile: Any) -> dict[str, Any]:
        """Format profile data for response.

        Args:
            profile: Raw profile data

        Returns:
            Formatted profile dictionary
        """
        formatted = {}

        # Extract organization info
        if hasattr(profile, "organization") and profile.organization:
            org = profile.organization
            formatted.update(
                {
                    "organization_name": getattr(org, "name", None),
                    "organization_type": getattr(org, "organization_type", None),
                    "billing_type": getattr(org, "billing_type", None),
                    "rate_limit_tier": getattr(org, "rate_limit_tier", None),
                }
            )

        # Extract account info
        if hasattr(profile, "account") and profile.account:
            acc = profile.account
            formatted.update(
                {
                    "email": getattr(acc, "email", None),
                    "full_name": getattr(acc, "full_name", None),
                    "display_name": getattr(acc, "display_name", None),
                    "has_claude_pro": getattr(acc, "has_claude_pro", None),
                    "has_claude_max": getattr(acc, "has_claude_max", None),
                }
            )

        # Remove None values
        return {k: v for k, v in formatted.items() if v is not None}

    async def get_auth_summary(self) -> dict[str, Any]:
        """Get authentication summary.

        Returns:
            Authentication status and details
        """
        if not self.credentials_manager:
            return {"auth": "not_configured"}

        try:
            if hasattr(self.credentials_manager, "get_auth_status"):
                auth_status = await self.credentials_manager.get_auth_status()

                summary = {"auth": "not_configured"}

                if auth_status.get("auth_configured"):
                    if auth_status.get("token_available"):
                        summary["auth"] = "authenticated"
                        if "time_remaining" in auth_status:
                            summary["auth_expires"] = auth_status["time_remaining"]
                        if "token_expired" in auth_status:
                            summary["auth_expired"] = auth_status["token_expired"]
                        if "subscription_type" in auth_status:
                            summary["subscription"] = auth_status["subscription_type"]
                    else:
                        summary["auth"] = "no_token"

                return summary

        except Exception as e:
            logger.warning(
                "auth_status_error",
                plugin=self.name,
                error=str(e),
                exc_info=e,
                category="plugin",
            )

        return {"auth": "status_error"}
