"""Codex/OpenAI OAuth provider for plugin registration."""

import hashlib
from base64 import urlsafe_b64encode
from typing import Any
from urllib.parse import urlencode

import httpx

from ccproxy.auth.oauth.protocol import ProfileLoggingMixin, StandardProfileFields
from ccproxy.auth.oauth.registry import OAuthProviderInfo
from ccproxy.core.logging import get_plugin_logger
from plugins.oauth_codex.client import CodexOAuthClient
from plugins.oauth_codex.config import CodexOAuthConfig
from plugins.oauth_codex.models import OpenAICredentials
from plugins.oauth_codex.storage import CodexTokenStorage


logger = get_plugin_logger()


class CodexOAuthProvider(ProfileLoggingMixin):
    """Codex/OpenAI OAuth provider implementation for registry."""

    def __init__(
        self,
        config: CodexOAuthConfig | None = None,
        storage: CodexTokenStorage | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
    ):
        """Initialize Codex OAuth provider.

        Args:
            config: OAuth configuration
            storage: Token storage
            http_client: Optional HTTP client (for request tracing support)
            hook_manager: Optional hook manager for emitting events
        """
        self.config = config or CodexOAuthConfig()
        self.storage = storage or CodexTokenStorage()
        self.hook_manager = hook_manager

        self.client = CodexOAuthClient(
            self.config, self.storage, http_client, hook_manager=hook_manager
        )

    @property
    def provider_name(self) -> str:
        """Internal provider name."""
        return "codex"

    @property
    def provider_display_name(self) -> str:
        """Display name for UI."""
        return "OpenAI Codex"

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
        return False  # OpenAI uses PKCE flow without client secret

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
            "redirect_uri": self.config.get_redirect_uri(),
            "response_type": "code",
            "scope": " ".join(self.config.scopes),
            "state": state,
            "audience": self.config.audience,
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
            "codex_oauth_auth_url_generated",
            state=state,
            has_pkce=bool(code_verifier and self.config.use_pkce),
            audience=self.config.audience,
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
            OpenAI credentials object
        """
        # Use the client's handle_callback method which includes code exchange
        credentials: OpenAICredentials = await self.client.handle_callback(
            code, state, code_verifier or ""
        )

        # The client already saves to storage if available, but we can save again
        # to our specific storage if needed
        if self.storage:
            await self.storage.save(credentials)

        logger.info(
            "codex_oauth_callback_handled",
            state=state,
            has_credentials=bool(credentials),
            has_id_token=bool(credentials.id_token),
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

        logger.info("codex_oauth_token_refreshed", category="auth")

        return credentials

    async def revoke_token(self, token: str) -> None:
        """Revoke an access or refresh token.

        Args:
            token: Token to revoke
        """
        # OpenAI doesn't have a revoke endpoint, so we just delete stored credentials
        if self.storage:
            await self.storage.delete()

        logger.info("codex_oauth_token_revoked_locally", category="auth")

    def get_provider_info(self) -> OAuthProviderInfo:
        """Get provider information for discovery.

        Returns:
            Provider information
        """
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="OAuth authentication for OpenAI Codex",
            supports_pkce=self.supports_pkce,
            scopes=self.config.scopes,
            is_available=True,
            plugin_name="oauth_codex",
        )

    async def validate_token(self, access_token: str) -> bool:
        """Validate an access token.

        Args:
            access_token: Token to validate

        Returns:
            True if token is valid
        """
        # OpenAI doesn't have a validation endpoint, so we check if stored token matches
        if self.storage:
            credentials = await self.storage.load()
            if credentials:
                return credentials.access_token == access_token
        return False

    async def get_user_info(self, access_token: str) -> dict[str, Any] | None:
        """Get user information using access token.

        Args:
            access_token: Valid access token

        Returns:
            User information or None
        """
        # Load stored credentials
        if self.storage:
            credentials = await self.storage.load()
            if credentials:
                info = {
                    "account_id": credentials.account_id,
                    "active": credentials.active,
                    "has_id_token": bool(credentials.id_token),
                }

                # Try to extract info from ID token if present
                if credentials.id_token:
                    try:
                        import jwt

                        decoded = jwt.decode(
                            credentials.id_token,
                            options={"verify_signature": False},
                        )
                        info.update(
                            {
                                "email": decoded.get("email"),
                                "name": decoded.get("name"),
                                "sub": decoded.get("sub"),
                            }
                        )
                    except Exception:
                        pass

                return info
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

    async def save_credentials(
        self, credentials: Any, custom_path: Any | None = None
    ) -> bool:
        """Save credentials using provider's storage mechanism.

        Args:
            credentials: OpenAI credentials object
            custom_path: Optional custom storage path (Path object)

        Returns:
            True if saved successfully, False otherwise
        """
        from pathlib import Path

        from ccproxy.auth.storage.generic import GenericJsonStorage
        from plugins.oauth_codex.manager import CodexTokenManager
        from plugins.oauth_codex.models import OpenAICredentials

        try:
            if custom_path:
                # Use custom path for storage
                storage = GenericJsonStorage(Path(custom_path), OpenAICredentials)
                manager = await CodexTokenManager.create(storage=storage)
            else:
                # Use default storage
                manager = await CodexTokenManager.create()

            return await manager.save_credentials(credentials)
        except Exception as e:
            logger.error(
                "Failed to save OpenAI credentials",
                error=str(e),
                exc_info=e,
                has_custom_path=bool(custom_path),
            )
            return False

    async def load_credentials(self, custom_path: Any | None = None) -> Any | None:
        """Load credentials from provider's storage.

        Args:
            custom_path: Optional custom storage path (Path object)

        Returns:
            Credentials if found, None otherwise
        """
        from pathlib import Path

        from ccproxy.auth.storage.generic import GenericJsonStorage
        from plugins.oauth_codex.manager import CodexTokenManager
        from plugins.oauth_codex.models import OpenAICredentials

        try:
            if custom_path:
                # Load from custom path
                storage = GenericJsonStorage(Path(custom_path), OpenAICredentials)
                manager = await CodexTokenManager.create(storage=storage)
            else:
                # Load from default storage
                manager = await CodexTokenManager.create()

            credentials = await manager.load_credentials()

            # Use standardized profile logging
            self._log_credentials_loaded("codex", credentials)

            return credentials
        except Exception as e:
            logger.error(
                "Failed to load OpenAI credentials",
                error=str(e),
                exc_info=e,
                has_custom_path=bool(custom_path),
            )
            return None

    def _extract_standard_profile(
        self, credentials: OpenAICredentials
    ) -> StandardProfileFields:
        """Extract standardized profile fields from OpenAI credentials for UI display.

        Args:
            credentials: OpenAI credentials with JWT tokens

        Returns:
            StandardProfileFields with clean, UI-friendly data
        """
        # Initialize with basic credential info
        from typing import Any

        profile_data: dict[str, Any] = {
            "account_id": credentials.account_id,
            "provider_type": "codex",
            "active": credentials.active,
            "expired": credentials.is_expired(),
            "has_refresh_token": bool(credentials.refresh_token),
            "has_id_token": bool(credentials.id_token),
            "token_expires_at": credentials.expires_at,
        }

        # Store raw credential data for debugging
        raw_data: dict[str, Any] = {
            "last_refresh": credentials.last_refresh,
            "expires_at": str(credentials.expires_at),
        }

        # Extract information from ID token
        if credentials.id_token:
            try:
                import jwt

                id_claims = jwt.decode(
                    credentials.id_token, options={"verify_signature": False}
                )

                # Extract UI-friendly profile info
                profile_data.update(
                    {
                        "email": id_claims.get("email"),
                        "email_verified": id_claims.get("email_verified"),
                        "display_name": id_claims.get("name")
                        or id_claims.get("given_name"),
                    }
                )

                # Extract subscription information
                auth_claims = id_claims.get("https://api.openai.com/auth", {})
                if isinstance(auth_claims, dict):
                    plan_type = auth_claims.get(
                        "chatgpt_plan_type"
                    )  # 'plus', 'pro', etc.
                    profile_data.update(
                        {
                            "subscription_type": plan_type,
                            "subscription_status": "active" if plan_type else None,
                        }
                    )

                    # Parse subscription dates
                    if auth_claims.get("chatgpt_subscription_active_until"):
                        try:
                            from datetime import datetime

                            expires_str = auth_claims[
                                "chatgpt_subscription_active_until"
                            ]
                            profile_data["subscription_expires_at"] = (
                                datetime.fromisoformat(
                                    expires_str.replace("+00:00", "")
                                )
                            )
                        except Exception:
                            pass

                    # Extract organization info
                    orgs = auth_claims.get("organizations", [])
                    if orgs:
                        primary_org = orgs[0] if isinstance(orgs, list) else {}
                        if isinstance(primary_org, dict):
                            profile_data.update(
                                {
                                    "organization_name": primary_org.get("title"),
                                    "organization_role": primary_org.get("role"),
                                }
                            )

                # Store full claims for debugging
                raw_data["id_token_claims"] = id_claims

            except Exception as e:
                logger.debug(
                    "Failed to decode ID token for profile extraction", error=str(e)
                )
                raw_data["id_token_decode_error"] = str(e)

        # Extract access token information
        if credentials.access_token:
            try:
                import jwt

                access_claims = jwt.decode(
                    credentials.access_token, options={"verify_signature": False}
                )

                # Store access token info in raw data
                raw_data["access_token_claims"] = {
                    "scopes": access_claims.get("scp", []),
                    "client_id": access_claims.get("client_id"),
                    "audience": access_claims.get("aud"),
                }

            except Exception as e:
                logger.debug(
                    "Failed to decode access token for profile extraction", error=str(e)
                )
                raw_data["access_token_decode_error"] = str(e)

        # Add provider-specific features
        if profile_data.get("subscription_type"):
            profile_data["features"] = {
                "chatgpt_plus": profile_data["subscription_type"] == "plus",
                "has_subscription": True,
            }

        profile_data["_raw_profile_data"] = raw_data

        return StandardProfileFields(**profile_data)

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.close()
