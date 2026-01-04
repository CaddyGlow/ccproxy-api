"""OAuth Claude plugin v2 implementation."""

from typing import Any, cast

from ccproxy.auth.oauth import OAuthProviderProtocol
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    AuthProviderPluginFactory,
    AuthProviderPluginRuntime,
    PluginContext,
    PluginManifest,
)

from .config import ClaudeOAuthConfig
from .provider import ClaudeOAuthProvider


logger = get_plugin_logger()


class OAuthClaudeRuntime(AuthProviderPluginRuntime):
    """Runtime for OAuth Claude plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: ClaudeOAuthConfig | None = None

    async def _on_initialize(self) -> None:
        """Initialize the OAuth Claude plugin."""
        logger.debug(
            "oauth_claude_initializing",
            context_keys=list(self.context.keys()) if self.context else [],
        )

        # Get configuration
        if self.context:
            config = self.context.get("config")
            if not isinstance(config, ClaudeOAuthConfig):
                # Use default config if none provided
                config = ClaudeOAuthConfig()
            logger.debug("oauth_claude_using_default_config")
            self.config = config

        # Call parent initialization which handles provider registration
        await super()._on_initialize()

        logger.debug(
            "oauth_claude_plugin_initialized",
            status="initialized",
            provider_name=self.auth_provider.provider_name
            if self.auth_provider
            else "unknown",
            category="plugin",
        )


class OAuthClaudeFactory(AuthProviderPluginFactory):
    """Factory for OAuth Claude plugin."""

    cli_safe = True  # Safe for CLI - provides auth only

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="oauth_claude",
            version="0.1.0",
            description="Standalone Claude OAuth authentication provider plugin",
            is_provider=True,  # It's a provider plugin but focused on OAuth
            config_class=ClaudeOAuthConfig,
            dependencies=[],
            routes=[],  # No HTTP routes needed
            tasks=[],  # No scheduled tasks needed
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        """Create context with auth provider components.

        Args:
            core_services: Core services container

        Returns:
            Plugin context with auth provider components
        """
        # Start with base context
        context = super().create_context(core_services)

        # Create auth provider for this plugin
        auth_provider = self.create_auth_provider(context)
        context["auth_provider"] = auth_provider

        # Add other auth-specific components if needed
        storage = self.create_storage()
        if storage:
            context["storage"] = storage

        return context

    def create_runtime(self) -> OAuthClaudeRuntime:
        """Create runtime instance."""
        return OAuthClaudeRuntime(self.manifest)

    def create_auth_provider(
        self, context: PluginContext | None = None
    ) -> OAuthProviderProtocol:
        """Create OAuth provider instance.

        Args:
            context: Plugin context containing shared resources

        Returns:
            ClaudeOAuthProvider instance
        """
        # Prefer validated config from context when available
        if context and isinstance(context.get("config"), ClaudeOAuthConfig):
            cfg = cast(ClaudeOAuthConfig, context.get("config"))
        else:
            cfg = ClaudeOAuthConfig()
        config: ClaudeOAuthConfig = cfg
        http_client = context.get("http_client") if context else None
        hook_manager = context.get("hook_manager") if context else None
        # CLIDetectionService is injected under 'cli_detection_service' in base context
        detection_service = context.get("cli_detection_service") if context else None
        settings = context.get("settings") if context else None
        provider = ClaudeOAuthProvider(
            config,
            http_client=http_client,
            hook_manager=hook_manager,
            detection_service=detection_service,
            settings=settings,
        )
        return cast(OAuthProviderProtocol, provider)

    def create_storage(self) -> Any | None:
        """Create storage for OAuth credentials.

        Returns:
            Storage instance or None to use provider's default
        """
        # ClaudeOAuthProvider manages its own storage internally
        return None


# Export the factory instance
factory = OAuthClaudeFactory()
