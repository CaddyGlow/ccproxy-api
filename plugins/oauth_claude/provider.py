"""Claude OAuth provider for plugin registration."""

import hashlib
from base64 import urlsafe_b64encode
from typing import Any
from urllib.parse import urlencode

from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.oauth.registry import OAuthProviderInfo
from ccproxy.core.logging import get_plugin_logger
from plugins.oauth_claude.client import ClaudeOAuthClient
from plugins.oauth_claude.config import ClaudeOAuthConfig
from plugins.oauth_claude.storage import ClaudeOAuthStorage


logger = get_plugin_logger()


class ClaudeOAuthProvider:
    """Claude OAuth provider implementation for registry."""

    def __init__(
        self,
        config: ClaudeOAuthConfig | None = None,
        storage: ClaudeOAuthStorage | None = None,
    ):
        """Initialize Claude OAuth provider.

        Args:
            config: OAuth configuration
            storage: Token storage
        """
        self.config = config or ClaudeOAuthConfig()
        self.storage = storage or ClaudeOAuthStorage()
        self.client = ClaudeOAuthClient(self.config, self.storage)

    @property
    def provider_name(self) -> str:
        """Internal provider name."""
        return "claude-api"

    @property
    def provider_display_name(self) -> str:
        """Display name for UI."""
        return "Claude API"

    @property
    def supports_pkce(self) -> bool:
        """Whether this provider supports PKCE."""
        return self.config.use_pkce

    @property
    def supports_refresh(self) -> bool:
        """Whether this provider supports token refresh."""
        return True

    @property
    def requires_client_secret(self) -> bool:
        """Whether this provider requires a client secret."""
        return False  # Claude uses PKCE-like flow without client secret

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
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
        }

        # Add PKCE challenge if supported and verifier provided
        if self.config.use_pkce and code_verifier:
            code_challenge = (
                urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
                .decode()
                .rstrip("=")
            )
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        auth_url = f"{self.config.authorize_url}?{urlencode(params)}"

        logger.info(
            "claude_oauth_auth_url_generated",
            state=state,
            has_pkce=bool(code_verifier and self.config.use_pkce),
            category="auth",
        )

        return auth_url

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        """Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from OAuth callback
            state: State parameter for validation
            code_verifier: PKCE code verifier (if PKCE is used)

        Returns:
            Claude credentials object
        """
        # Use the client's handle_callback method which includes code exchange
        credentials: ClaudeCredentials = await self.client.handle_callback(
            code, state, code_verifier or ""
        )

        # The client already saves to storage if available, but we can save again
        # to our specific storage if needed
        if self.storage:
            await self.storage.save(credentials)

        logger.info(
            "claude_oauth_callback_handled",
            state=state,
            has_credentials=bool(credentials),
            category="auth",
        )

        return credentials

    async def refresh_access_token(self, refresh_token: str) -> Any:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token from previous auth

        Returns:
            New token response
        """
        credentials = await self.client.refresh_token(refresh_token)

        # Store updated credentials
        if self.storage:
            await self.storage.save(credentials)

        logger.info("claude_oauth_token_refreshed", category="auth")

        return credentials

    async def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token.

        Args:
            token: Token to revoke
        """
        # Claude doesn't have a revoke endpoint, so we just delete stored credentials
        if self.storage:
            await self.storage.delete()

        logger.info("claude_oauth_token_revoked_locally", category="auth")

    def get_provider_info(self) -> OAuthProviderInfo:
        """Get provider information for discovery.

        Returns:
            Provider information
        """
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="OAuth authentication for Claude AI",
            supports_pkce=self.supports_pkce,
            scopes=self.config.scopes,
            is_available=True,
            plugin_name="oauth_claude",
        )

    async def validate_token(self, access_token: str) -> bool:
        """Validate an access token.

        Args:
            access_token: Token to validate

        Returns:
            True if token is valid
        """
        # Claude doesn't have a validation endpoint, so we check if stored token matches
        if self.storage:
            credentials = await self.storage.load()
            if credentials and credentials.claude_ai_oauth:
                stored_token = (
                    credentials.claude_ai_oauth.access_token.get_secret_value()
                )
                return stored_token == access_token
        return False

    async def get_user_info(self, access_token: str) -> dict[str, Any] | None:
        """Get user information using access token.

        Args:
            access_token: Valid access token

        Returns:
            User information or None
        """
        # Load stored credentials which contain user info
        if self.storage:
            credentials = await self.storage.load()
            if credentials and credentials.claude_ai_oauth:
                return {
                    "subscription_type": credentials.claude_ai_oauth.subscription_type,
                    "scopes": credentials.claude_ai_oauth.scopes,
                    "token_type": credentials.claude_ai_oauth.token_type,
                }
        return None

    def get_storage(self) -> Any:
        """Get storage implementation for this provider.

        Returns:
            Storage implementation
        """
        return self.storage

    def get_config(self) -> Any:
        """Get configuration for this provider.

        Returns:
            Configuration implementation
        """
        return self.config

    def get_credential_summary(self, credentials: ClaudeCredentials) -> dict[str, Any]:
        """Get a summary of credentials for display.

        Args:
            credentials: Claude credentials

        Returns:
            Dictionary with display-friendly credential summary
        """
        summary = {
            "provider": self.provider_display_name,
            "authenticated": bool(credentials and credentials.claude_ai_oauth),
        }

        if credentials and credentials.claude_ai_oauth:
            oauth = credentials.claude_ai_oauth
            summary.update(
                {
                    "subscription_type": oauth.subscription_type,
                    "scopes": oauth.scopes,
                    "has_refresh_token": bool(oauth.refresh_token),
                    "expired": oauth.is_expired
                    if hasattr(oauth, "is_expired")
                    else False,
                }
            )

            # Add expiration info if available
            if hasattr(oauth, "expires_at_datetime"):
                summary["expires_at"] = oauth.expires_at_datetime.isoformat()

        return summary
