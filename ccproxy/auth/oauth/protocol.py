"""OAuth protocol definitions for plugin OAuth implementations.

This module defines the protocols and interfaces that plugins must implement
to provide OAuth authentication capabilities.
"""

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field

from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


# Import CLI types from registry to avoid duplication


class StandardProfileFields(BaseModel):
    """Standardized profile fields for consistent UI display across OAuth providers."""

    # Core Identity
    account_id: str
    provider_type: str  # 'claude', 'codex', etc.
    email: str | None = None
    display_name: str | None = None

    # Account Status
    authenticated: bool = True
    active: bool = True
    expired: bool = False

    # Subscription/Plan Information
    subscription_type: str | None = None  # 'plus', 'pro', 'max', 'free'
    subscription_status: str | None = None  # 'active', 'expired', 'cancelled'
    subscription_expires_at: datetime | None = None

    # Token Information
    has_refresh_token: bool = False
    has_id_token: bool = False
    token_expires_at: datetime | None = None

    # Organization/Team
    organization_name: str | None = None
    organization_role: str | None = None  # 'owner', 'admin', 'member'

    # Verification Status
    email_verified: bool | None = None

    # Additional Features (provider-specific)
    features: dict[str, Any] = Field(
        default_factory=dict
    )  # For provider-specific features like 'has_claude_max'

    # Raw data (for debugging, not UI display)
    raw_profile_data: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,  # Exclude raw data from normal serialization
    )


class ProfileLoggingMixin:
    """Mixin to provide standardized profile dump logging for OAuth providers."""

    def _log_profile_dump(
        self, provider_name: str, profile: StandardProfileFields, category: str = "auth"
    ) -> None:
        """Log standardized profile data in UI-friendly format.

        Args:
            provider_name: Name of the OAuth provider (e.g., 'claude', 'codex')
            profile: Standardized profile fields for UI display
            category: Log category (defaults to 'auth')
        """
        # Log clean UI-friendly profile data
        profile_data = profile.model_dump(exclude={"raw_profile_data"})
        logger.debug(
            f"{provider_name}_profile_full_dump",
            profile_data=profile_data,
            category=category,
        )

        # Optionally log raw data separately for debugging (only if needed)
        if profile.raw_profile_data:
            logger.debug(
                f"{provider_name}_profile_raw_data",
                raw_data=profile.raw_profile_data,
                category="auth_debug",
            )

    @abstractmethod
    def _extract_standard_profile(self, credentials: Any) -> StandardProfileFields:
        """Extract standardized profile fields from provider-specific credentials.

        This method should be implemented by each OAuth provider to map their
        credential format to the standardized profile fields for UI display.

        Args:
            credentials: Provider-specific credentials object

        Returns:
            StandardProfileFields with clean, UI-friendly data
        """
        pass

    async def get_standard_profile(
        self, credentials: Any | None = None
    ) -> StandardProfileFields | None:
        """Return standardized profile fields for UI display.

        If credentials are not provided, attempts to load them via a
        provider's `load_credentials()` method when available. This method
        intentionally avoids network calls and relies on locally available
        information or cached profile data inside provider implementations.

        Args:
            credentials: Optional provider-specific credentials

        Returns:
            StandardProfileFields or None if unavailable
        """
        try:
            creds = credentials
            if creds is None and hasattr(self, "load_credentials"):
                # Best-effort local load (provider-specific, may use storage)
                load_fn = self.load_credentials
                if callable(load_fn):
                    creds = await cast(Callable[[], Awaitable[Any]], load_fn)()

            if not creds:
                return None

            return self._extract_standard_profile(creds)
        except Exception as e:
            logger.debug(
                "standard_profile_generation_failed",
                provider=getattr(self, "provider_name", type(self).__name__),
                error=str(e),
            )
            return None

    def _log_credentials_loaded(
        self, provider_name: str, credentials: Any, category: str = "auth"
    ) -> None:
        """Log credentials loaded with standardized profile data.

        Args:
            provider_name: Name of the OAuth provider
            credentials: Loaded credentials object
            category: Log category
        """
        if credentials:
            try:
                profile = self._extract_standard_profile(credentials)
                self._log_profile_dump(provider_name, profile, category)
            except Exception as e:
                logger.debug(
                    f"{provider_name}_profile_extraction_failed",
                    error=str(e),
                    category=category,
                )


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


# Import the full protocol from registry


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
