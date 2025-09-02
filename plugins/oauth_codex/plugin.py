"""OAuth Codex plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    AuthProviderPluginFactory,
    AuthProviderPluginRuntime,
    PluginContext,
    PluginManifest,
)

from .config import CodexOAuthConfig
from .provider import CodexOAuthProvider


logger = get_plugin_logger()


class OAuthCodexRuntime(AuthProviderPluginRuntime):
    """Runtime for OAuth Codex plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: CodexOAuthConfig | None = None

    async def _on_initialize(self) -> None:
        """Initialize the OAuth Codex plugin."""
        logger.debug(
            "oauth_codex_initializing",
            context_keys=list(self.context.keys()) if self.context else [],
        )

        # Get configuration
        if self.context:
            config = self.context.get("config")
            if not isinstance(config, CodexOAuthConfig):
                # Use default config if none provided
                config = CodexOAuthConfig()
                logger.info("oauth_codex_using_default_config")
            self.config = config

        # Call parent initialization which handles provider registration
        await super()._on_initialize()

        logger.debug(
            "oauth_codex_plugin_initialized",
            status="initialized",
            provider_name=self.auth_provider.provider_name
            if self.auth_provider
            else "unknown",
            category="plugin",
        )


class OAuthCodexFactory(AuthProviderPluginFactory):
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

    def create_auth_provider(
        self, context: PluginContext | None = None
    ) -> CodexOAuthProvider:
        """Create OAuth provider instance.

        Args:
            context: Optional plugin context containing http_client

        Returns:
            CodexOAuthProvider instance
        """
        # Prefer validated config from context when available
        config = (
            context.get("config")
            if context and isinstance(context.get("config"), CodexOAuthConfig)
            else CodexOAuthConfig()
        )
        http_client = context.get("http_client") if context else None
        hook_manager = context.get("hook_manager") if context else None
        return CodexOAuthProvider(
            config, http_client=http_client, hook_manager=hook_manager
        )

    def create_storage(self) -> Any | None:
        """Create storage for OAuth credentials.

        Returns:
            Storage instance or None to use provider's default
        """
        # CodexOAuthProvider manages its own storage internally
        return None


# Export the factory instance
factory = OAuthCodexFactory()
