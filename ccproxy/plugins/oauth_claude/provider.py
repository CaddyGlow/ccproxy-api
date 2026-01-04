"""Claude OAuth provider for plugin registration."""

import hashlib
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

from ccproxy.auth.oauth.protocol import ProfileLoggingMixin, StandardProfileFields
from ccproxy.auth.oauth.registry import CliAuthConfig, FlowType, OAuthProviderInfo
from ccproxy.auth.storage.generic import GenericJsonStorage
from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    from ccproxy.services.cli_detection import CLIDetectionService

    from .manager import ClaudeApiTokenManager

from ccproxy.core.logging import get_plugin_logger

from .client import ClaudeOAuthClient
from .config import ClaudeOAuthConfig
from .models import ClaudeCredentials, ClaudeProfileInfo
from .storage import ClaudeOAuthStorage


logger = get_plugin_logger()


class ClaudeOAuthProvider(ProfileLoggingMixin):
    """Claude OAuth provider implementation for registry."""

    def __init__(
        self,
        config: ClaudeOAuthConfig | None = None,
        storage: ClaudeOAuthStorage | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: "CLIDetectionService | None" = None,
        settings: Settings | None = None,
    ):
        """Initialize Claude OAuth provider.

        Args:
            config: OAuth configuration
            storage: Token storage
            http_client: Optional HTTP client (for request tracing support)
            hook_manager: Optional hook manager for emitting events
            detection_service: Optional CLI detection service for headers
            settings: Optional settings for HTTP client configuration
        """
        self.config = config or ClaudeOAuthConfig()
        self.storage = storage or ClaudeOAuthStorage()
        self.hook_manager = hook_manager
        self.detection_service = detection_service
        self.http_client = http_client
        self.settings = settings
        self._cached_profile: ClaudeProfileInfo | None = (
            None  # Cache enhanced profile data for UI display
        )

        self.client = ClaudeOAuthClient(
            self.config,
            self.storage,
            http_client,
            hook_manager=hook_manager,
            detection_service=detection_service,
            settings=settings,
        )

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
        self,
        state: str,
        code_verifier: str | None = None,
        redirect_uri: str | None = None,
    ) -> str:
        """Get the authorization URL for OAuth flow.

        Args:
            state: OAuth state parameter for CSRF protection
            code_verifier: PKCE code verifier (if PKCE is supported)

        Returns:
            Authorization URL to redirect user to
        """
        # Use provided redirect URI or fall back to config default
        if redirect_uri is None:
            redirect_uri = self.config.get_redirect_uri()

        params = {
            "code": "true",  # Required by Claude OAuth
            "client_id": self.config.client_id,
            "redirect_uri": redirect_uri,
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
        self,
        code: str,
        state: str,
        code_verifier: str | None = None,
        redirect_uri: str | None = None,
    ) -> Any:
        """Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from OAuth callback
            state: State parameter for validation
            code_verifier: PKCE code verifier (if PKCE is used)
            redirect_uri: Redirect URI used in authorization (optional)

        Returns:
            Claude credentials object
        """
        # Use the client's handle_callback method which includes code exchange
        # If a specific redirect_uri was provided, create a temporary client with that URI
        if redirect_uri and redirect_uri != self.client.redirect_uri:
            # Create temporary config with the specific redirect URI
            temp_config = ClaudeOAuthConfig(
                client_id=self.config.client_id,
                redirect_uri=redirect_uri,
                scopes=self.config.scopes,
                base_url=self.config.base_url,
                authorize_url=self.config.authorize_url,
                token_url=self.config.token_url,
                use_pkce=self.config.use_pkce,
            )

            # Create temporary client with the correct redirect URI
            temp_client = ClaudeOAuthClient(
                temp_config,
                self.storage,
                self.http_client,
                hook_manager=self.hook_manager,
                detection_service=self.detection_service,
                settings=self.settings,
            )

            credentials = await temp_client.handle_callback(
                code, state, code_verifier or ""
            )
        else:
            # Use the regular client
            credentials = await self.client.handle_callback(
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

    async def save_credentials(
        self, credentials: Any, custom_path: Any | None = None
    ) -> bool:
        """Save credentials using provider's storage mechanism.

        Args:
            credentials: Claude credentials object
            custom_path: Optional custom storage path (Path object)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if custom_path:
                # Use custom path for storage
                storage = GenericJsonStorage(Path(custom_path), ClaudeCredentials)
                manager = await self.create_token_manager(storage=storage)
            else:
                # Use default storage
                manager = await self.create_token_manager()

            return await manager.save_credentials(credentials)
        except Exception as e:
            logger.error(
                "Failed to save Claude credentials",
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
        try:
            if custom_path:
                # Load from custom path
                storage = GenericJsonStorage(Path(custom_path), ClaudeCredentials)
                manager = await self.create_token_manager(storage=storage)
            else:
                # Load from default storage
                manager = await self.create_token_manager()

            credentials = await manager.load_credentials()

            # Use standardized profile logging with rich Claude profile data
            if credentials:
                profile = await manager.get_profile()
                if profile:
                    # Cache profile for UI display
                    self._cached_profile = profile
                    # Create enhanced standardized profile with rich Claude data
                    standard_profile = self._create_enhanced_profile(
                        credentials, profile
                    )
                    self._log_profile_dump("claude", standard_profile)

            return credentials
        except Exception as e:
            logger.error(
                "Failed to load Claude credentials",
                error=str(e),
                exc_info=e,
                has_custom_path=bool(custom_path),
            )
            return None

    async def create_token_manager(
        self, storage: Any | None = None
    ) -> "ClaudeApiTokenManager":
        """Create token manager with proper dependency injection.

        Provided to allow core/CLI code to obtain a manager without
        importing plugin classes directly.
        """
        from .manager import ClaudeApiTokenManager

        return await ClaudeApiTokenManager.create(
            storage=storage,
            http_client=self.http_client,
            oauth_provider=self,  # Inject self as protocol
        )

    def _extract_standard_profile(
        self, credentials: ClaudeCredentials
    ) -> StandardProfileFields:
        """Extract standardized profile fields from Claude credentials for UI display.

        Args:
            credentials: Claude credentials with profile information

        Returns:
            StandardProfileFields with clean, UI-friendly data
        """
        # Use cached enhanced profile data if available
        if self._cached_profile:
            return self._create_enhanced_profile(credentials, self._cached_profile)

        # Fallback to basic credential info
        from typing import Any

        profile_data: dict[str, Any] = {
            "account_id": getattr(credentials, "account_id", "unknown"),
            "provider_type": "claude-api",
            "active": getattr(credentials, "active", True),
            "expired": False,  # Claude handles expiration internally
            "has_refresh_token": bool(getattr(credentials, "refresh_token", None)),
        }

        # Store raw credential data for debugging
        raw_data = {}
        if hasattr(credentials, "model_dump"):
            raw_data["credentials"] = credentials.model_dump()

        profile_data["raw_profile_data"] = raw_data

        return StandardProfileFields(**profile_data)

    def _create_enhanced_profile(
        self, credentials: ClaudeCredentials, profile: Any
    ) -> StandardProfileFields:
        """Create enhanced standardized profile with rich Claude profile data.

        Args:
            credentials: Claude credentials
            profile: Rich profile data from manager

        Returns:
            StandardProfileFields with full Claude profile information
        """
        # Create basic profile data without recursion
        basic_profile_data: dict[str, Any] = {
            "account_id": getattr(credentials, "account_id", "unknown"),
            "provider_type": "claude-api",
            "active": getattr(credentials, "active", True),
            "expired": False,  # Claude handles expiration internally
            "has_refresh_token": bool(getattr(credentials, "refresh_token", None)),
            "raw_profile_data": {},
        }

        # Extract profile data
        profile_dict = (
            profile.model_dump()
            if hasattr(profile, "model_dump")
            else {"profile": str(profile)}
        )

        # Map Claude profile fields to standard fields
        updates = {}

        if profile_dict.get("account_id"):
            updates["account_id"] = profile_dict["account_id"]

        if profile_dict.get("email"):
            updates["email"] = profile_dict["email"]

        if profile_dict.get("display_name"):
            updates["display_name"] = profile_dict["display_name"]

        # Extract subscription information from extras
        extras = profile_dict.get("extras", {})
        if isinstance(extras, dict):
            account = extras.get("account", {})
            if isinstance(account, dict):
                # Map Claude subscription types
                if account.get("has_claude_max"):
                    updates.update(
                        {
                            "subscription_type": "max",
                            "subscription_status": "active",
                        }
                    )
                elif account.get("has_claude_pro"):
                    updates.update(
                        {
                            "subscription_type": "pro",
                            "subscription_status": "active",
                        }
                    )

                # Features
                updates["features"] = {
                    "claude_max": account.get("has_claude_max", False),
                    "claude_pro": account.get("has_claude_pro", False),
                }

            # Organization info
            org = extras.get("organization", {})
            if isinstance(org, dict):
                updates.update(
                    {
                        "organization_name": org.get("name"),
                        "organization_role": "member",  # Claude doesn't provide role details
                    }
                )

        # Store full profile data in raw data (start from basic profile data)
        from typing import cast

        base_raw = cast(dict[str, Any], basic_profile_data.get("raw_profile_data", {}))
        raw_data = dict(base_raw)
        raw_data["full_profile"] = profile_dict
        updates["raw_profile_data"] = raw_data

        # Create new profile with updates starting from basic profile data
        profile_data = dict(basic_profile_data)
        profile_data.update(updates)

        return StandardProfileFields(**profile_data)

    async def exchange_manual_code(self, code: str) -> Any:
        """Exchange manual authorization code for tokens.

        Args:
            code: Authorization code from manual entry

        Returns:
            Claude credentials object
        """
        # For manual code flow, use OOB redirect URI and no state validation
        credentials: ClaudeCredentials = await self.client.handle_callback(
            code, "manual", ""
        )

        if self.storage:
            await self.storage.save(credentials)

        logger.info(
            "claude_oauth_manual_code_exchanged",
            has_credentials=bool(credentials),
            category="auth",
        )

        return credentials

    @property
    def cli(self) -> CliAuthConfig:
        """Get CLI authentication configuration for this provider."""
        return CliAuthConfig(
            preferred_flow=FlowType.browser,
            callback_port=54545,
            callback_path="/callback",
            supports_manual_code=True,
            supports_device_flow=False,
            fixed_redirect_uri=None,
            manual_redirect_uri="https://console.anthropic.com/oauth/code/callback",
        )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.client:
            await self.client.close()
