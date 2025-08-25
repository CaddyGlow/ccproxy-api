"""OAuth Codex plugin v2 implementation."""

from typing import Any

from ccproxy.auth.oauth.registry import get_oauth_registry
from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    ProviderPluginFactory,
    ProviderPluginRuntime,
)
from plugins.oauth_codex.config import CodexOAuthConfig
from plugins.oauth_codex.provider import CodexOAuthProvider


logger = get_plugin_logger()


class OAuthCodexRuntime(ProviderPluginRuntime):
    """Runtime for OAuth Codex plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: CodexOAuthConfig | None = None
        self.oauth_provider: CodexOAuthProvider | None = None

    async def _on_initialize(self) -> None:
        """Initialize the OAuth Codex plugin."""
        logger.debug(
            "oauth_codex_initializing",
            context_keys=list(self.context.keys()) if self.context else [],
        )

        await super()._on_initialize()

        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, CodexOAuthConfig):
            # Use default config if none provided
            config = CodexOAuthConfig()
            logger.info("oauth_codex_using_default_config")
        self.config = config

        # Create and register OAuth provider
        self.oauth_provider = CodexOAuthProvider(config)
        registry = get_oauth_registry()
        registry.register_provider(self.oauth_provider)

        logger.debug(
            "oauth_codex_plugin_initialized",
            status="initialized",
            provider_name=self.oauth_provider.provider_name,
            category="plugin",
        )

    async def _on_shutdown(self) -> None:
        """Shutdown the OAuth Codex plugin."""
        # Unregister OAuth provider
        if self.oauth_provider:
            registry = get_oauth_registry()
            registry.unregister_provider(self.oauth_provider.provider_name)
            logger.info(
                "oauth_codex_provider_unregistered",
                provider_name=self.oauth_provider.provider_name,
                category="plugin",
            )

        await super()._on_shutdown()

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = await super()._get_health_details()

        if self.oauth_provider:
            # Check if provider is registered
            registry = get_oauth_registry()
            is_registered = registry.has_provider(self.oauth_provider.provider_name)
            details.update(
                {
                    "oauth_provider_registered": is_registered,
                    "oauth_provider_name": self.oauth_provider.provider_name,
                }
            )

        return details


class OAuthCodexFactory(ProviderPluginFactory):
    """Factory for OAuth Codex plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="oauth_codex",
            version="1.0.0",
            description="Standalone OpenAI Codex OAuth authentication provider plugin",
            is_provider=True,  # It's a provider plugin but focused on OAuth
            config_class=CodexOAuthConfig,
            dependencies=[],
            routes=[],  # No HTTP routes needed
            tasks=[],  # No scheduled tasks needed
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> OAuthCodexRuntime:
        """Create runtime instance."""
        return OAuthCodexRuntime(self.manifest)

    def create_adapter(self, context: PluginContext) -> Any:
        """OAuth plugins don't need adapters.

        Args:
            context: Plugin context

        Returns:
            None as OAuth plugins don't proxy requests
        """
        return None

    def create_detection_service(self, context: PluginContext) -> Any:
        """OAuth plugins don't need detection services.

        Args:
            context: Plugin context

        Returns:
            None as OAuth plugins don't detect capabilities
        """
        return None

    def create_credentials_manager(self, context: PluginContext) -> Any:
        """OAuth plugins provide their own storage.

        Args:
            context: Plugin context

        Returns:
            None as OAuth providers manage their own storage
        """
        return None


# Export the factory instance
factory = OAuthCodexFactory()
