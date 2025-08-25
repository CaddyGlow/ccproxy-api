"""OAuth protocol definitions for plugin OAuth implementations.

This module defines the protocols and interfaces that plugins must implement
to provide OAuth authentication capabilities.
"""

from typing import Any, Protocol

from pydantic import BaseModel


class OAuthConfig(BaseModel):
    """Base configuration for OAuth providers."""

    client_id: str
    client_secret: str | None = None  # Not needed for PKCE flows
    redirect_uri: str
    authorize_url: str
    token_url: str
    scopes: list[str] = []
    use_pkce: bool = True


class OAuthStorageProtocol(Protocol):
    """Protocol for OAuth token storage implementations."""

    async def save_tokens(
        self,
        provider: str,
        access_token: str,
        refresh_token: str | None = None,
        expires_in: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Save OAuth tokens.

        Args:
            provider: Provider name
            access_token: Access token
            refresh_token: Optional refresh token
            expires_in: Token expiration in seconds
            **kwargs: Additional provider-specific data
        """
        ...

    async def get_tokens(self, provider: str) -> dict[str, Any] | None:
        """Retrieve stored tokens for a provider.

        Args:
            provider: Provider name

        Returns:
            Token data or None if not found
        """
        ...

    async def delete_tokens(self, provider: str) -> None:
        """Delete stored tokens for a provider.

        Args:
            provider: Provider name
        """
        ...

    async def has_valid_tokens(self, provider: str) -> bool:
        """Check if valid tokens exist for a provider.

        Args:
            provider: Provider name

        Returns:
            True if valid tokens exist
        """
        ...


class OAuthConfigProtocol(Protocol):
    """Protocol for OAuth configuration providers."""

    def get_client_id(self) -> str:
        """Get OAuth client ID."""
        ...

    def get_client_secret(self) -> str | None:
        """Get OAuth client secret (if applicable)."""
        ...

    def get_redirect_uri(self) -> str:
        """Get OAuth redirect URI."""
        ...

    def get_authorize_url(self) -> str:
        """Get authorization endpoint URL."""
        ...

    def get_token_url(self) -> str:
        """Get token endpoint URL."""
        ...

    def get_scopes(self) -> list[str]:
        """Get requested OAuth scopes."""
        ...

    def uses_pkce(self) -> bool:
        """Check if PKCE should be used."""
        ...


class TokenResponse(BaseModel):
    """Standard OAuth token response."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None

    # Additional fields that providers might include
    id_token: str | None = None  # For OpenID Connect
    account_id: str | None = None  # Provider-specific user ID


class OAuthProviderBase(Protocol):
    """Extended protocol for OAuth providers with additional capabilities."""

    @property
    def provider_name(self) -> str:
        """Internal provider name."""
        ...

    @property
    def provider_display_name(self) -> str:
        """Display name for UI."""
        ...

    @property
    def supports_pkce(self) -> bool:
        """Whether this provider supports PKCE."""
        ...

    @property
    def supports_refresh(self) -> bool:
        """Whether this provider supports token refresh."""
        ...

    @property
    def requires_client_secret(self) -> bool:
        """Whether this provider requires a client secret."""
        ...

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        """Get authorization URL."""
        ...

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        """Handle OAuth callback."""
        ...

    async def refresh_access_token(self, refresh_token: str) -> Any:
        """Refresh access token."""
        ...

    async def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        ...

    async def validate_token(self, access_token: str) -> bool:
        """Validate an access token.

        Args:
            access_token: Token to validate

        Returns:
            True if token is valid
        """
        ...

    async def get_user_info(self, access_token: str) -> dict[str, Any] | None:
        """Get user information using access token.

        Args:
            access_token: Valid access token

        Returns:
            User information or None
        """
        ...

    def get_storage(self) -> OAuthStorageProtocol | None:
        """Get storage implementation for this provider.

        Returns:
            Storage implementation or None if provider handles storage
        """
        ...

    def get_config(self) -> OAuthConfigProtocol | None:
        """Get configuration for this provider.

        Returns:
            Configuration implementation or None
        """
        ...
