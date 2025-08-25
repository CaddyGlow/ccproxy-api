"""OAuth Provider Registry for dynamic provider management.

This module provides a central registry where plugins can register their OAuth
providers at runtime, enabling dynamic discovery and management of OAuth flows.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from ccproxy.core.logging import get_logger


logger = get_logger()


class OAuthProviderInfo(BaseModel):
    """Information about a registered OAuth provider."""

    name: str
    display_name: str
    description: str = ""
    supports_pkce: bool = True
    scopes: list[str] = []
    is_available: bool = True
    plugin_name: str = ""


class OAuthProviderProtocol(Protocol):
    """Protocol for OAuth provider implementations."""

    @property
    def provider_name(self) -> str:
        """Internal provider name (e.g., 'claude-api', 'codex')."""
        ...

    @property
    def provider_display_name(self) -> str:
        """Display name for UI (e.g., 'Claude API', 'OpenAI Codex')."""
        ...

    @property
    def supports_pkce(self) -> bool:
        """Whether this provider supports PKCE flow."""
        ...

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        """Get the authorization URL for OAuth flow.

        Args:
            state: OAuth state parameter for CSRF protection
            code_verifier: PKCE code verifier (if PKCE is supported)

        Returns:
            Authorization URL to redirect user to
        """
        ...

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        """Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from OAuth callback
            state: State parameter for validation
            code_verifier: PKCE code verifier (if PKCE is used)

        Returns:
            Provider-specific credentials object
        """
        ...

    async def refresh_access_token(self, refresh_token: str) -> Any:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous auth

        Returns:
            New token response
        """
        ...

    async def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token.

        Args:
            token: Token to revoke
        """
        ...

    def get_provider_info(self) -> OAuthProviderInfo:
        """Get provider information for discovery.

        Returns:
            Provider information
        """
        ...

    @property
    def supports_refresh(self) -> bool:
        """Whether this provider supports token refresh."""
        ...

    def get_storage(self) -> Any:
        """Get storage implementation for this provider.

        Returns:
            Storage implementation or None
        """
        ...

    def get_credential_summary(self, credentials: Any) -> dict[str, Any]:
        """Get a summary of credentials for display.

        Args:
            credentials: Provider-specific credentials

        Returns:
            Dictionary with display-friendly credential summary
        """
        ...


class OAuthRegistry:
    """Central registry for OAuth providers.

    This registry allows plugins to register their OAuth providers at runtime,
    enabling dynamic discovery and management of OAuth authentication flows.
    """

    def __init__(self) -> None:
        """Initialize the OAuth registry."""
        self._providers: dict[str, OAuthProviderProtocol] = {}
        self._provider_info_cache: dict[str, OAuthProviderInfo] = {}
        logger.info("oauth_registry_initialized", category="auth")

    def register_provider(self, provider: OAuthProviderProtocol) -> None:
        """Register an OAuth provider from a plugin.

        Args:
            provider: OAuth provider implementation

        Raises:
            ValueError: If provider with same name already registered
        """
        provider_name = provider.provider_name

        if provider_name in self._providers:
            raise ValueError(f"OAuth provider '{provider_name}' is already registered")

        self._providers[provider_name] = provider

        # Cache provider info
        try:
            info = provider.get_provider_info()
            self._provider_info_cache[provider_name] = info
            logger.info(
                "oauth_provider_registered",
                provider=provider_name,
                display_name=info.display_name,
                supports_pkce=info.supports_pkce,
                plugin=info.plugin_name,
                category="auth",
            )
        except Exception as e:
            logger.error(
                "oauth_provider_info_error",
                provider=provider_name,
                error=str(e),
                exc_info=e,
                category="auth",
            )

    def unregister_provider(self, provider_name: str) -> None:
        """Unregister an OAuth provider.

        Args:
            provider_name: Name of provider to unregister
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            if provider_name in self._provider_info_cache:
                del self._provider_info_cache[provider_name]
            logger.info(
                "oauth_provider_unregistered", provider=provider_name, category="auth"
            )

    def get_provider(self, provider_name: str) -> OAuthProviderProtocol | None:
        """Get a registered OAuth provider by name.

        Args:
            provider_name: Name of the provider

        Returns:
            OAuth provider instance or None if not found
        """
        return self._providers.get(provider_name)

    def list_providers(self) -> dict[str, OAuthProviderInfo]:
        """List all registered OAuth providers.

        Returns:
            Dictionary mapping provider names to their info
        """
        result = {}
        for name, provider in self._providers.items():
            # Try to get fresh info, fall back to cache
            try:
                info = provider.get_provider_info()
                self._provider_info_cache[name] = info
                result[name] = info
            except Exception as e:
                logger.warning(
                    "oauth_provider_info_refresh_error",
                    provider=name,
                    error=str(e),
                    category="auth",
                )
                # Use cached info if available
                if name in self._provider_info_cache:
                    result[name] = self._provider_info_cache[name]

        return result

    def has_provider(self, provider_name: str) -> bool:
        """Check if a provider is registered.

        Args:
            provider_name: Name of the provider

        Returns:
            True if provider is registered
        """
        return provider_name in self._providers

    def get_provider_info(self, provider_name: str) -> OAuthProviderInfo | None:
        """Get information about a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider information or None if not found
        """
        provider = self.get_provider(provider_name)
        if not provider:
            return None

        try:
            info = provider.get_provider_info()
            self._provider_info_cache[provider_name] = info
            return info
        except Exception as e:
            logger.error(
                "oauth_provider_info_error",
                provider=provider_name,
                error=str(e),
                exc_info=e,
                category="auth",
            )
            # Return cached info if available
            return self._provider_info_cache.get(provider_name)

    def clear(self) -> None:
        """Clear all registered providers.

        This is mainly useful for testing or shutdown scenarios.
        """
        self._providers.clear()
        self._provider_info_cache.clear()
        logger.info("oauth_registry_cleared", category="auth")


# Global registry instance
_registry: OAuthRegistry | None = None


def get_oauth_registry() -> OAuthRegistry:
    """Get the global OAuth registry instance.

    Returns:
        Global OAuth registry
    """
    global _registry
    if _registry is None:
        _registry = OAuthRegistry()
    return _registry


def reset_oauth_registry() -> None:
    """Reset the global OAuth registry.

    This clears the existing registry and creates a new one.
    Mainly useful for testing.
    """
    global _registry
    if _registry:
        _registry.clear()
    _registry = OAuthRegistry()
