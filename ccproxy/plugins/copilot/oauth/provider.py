"""OAuth provider implementation for GitHub Copilot."""

import contextlib
from typing import TYPE_CHECKING, Any

import httpx

from ccproxy.auth.oauth.protocol import ProfileLoggingMixin, StandardProfileFields
from ccproxy.auth.oauth.registry import CliAuthConfig, FlowType, OAuthProviderInfo
from ccproxy.core.logging import get_plugin_logger

from ..config import CopilotOAuthConfig
from .client import CopilotOAuthClient
from .models import CopilotCredentials, CopilotProfileInfo, CopilotTokenInfo
from .storage import CopilotOAuthStorage


if TYPE_CHECKING:
    from ccproxy.services.cli_detection import CLIDetectionService


logger = get_plugin_logger()


class CopilotOAuthProvider(ProfileLoggingMixin):
    """GitHub Copilot OAuth provider implementation."""

    def __init__(
        self,
        config: CopilotOAuthConfig | None = None,
        storage: CopilotOAuthStorage | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: "CLIDetectionService | None" = None,
    ):
        """Initialize Copilot OAuth provider.

        Args:
            config: OAuth configuration
            storage: Token storage
            http_client: Optional HTTP client for request tracing
            hook_manager: Optional hook manager for events
            detection_service: Optional CLI detection service
        """
        self.config = config or CopilotOAuthConfig()
        self.storage = storage or CopilotOAuthStorage()
        self.hook_manager = hook_manager
        self.detection_service = detection_service
        self.http_client = http_client
        self._cached_profile: CopilotProfileInfo | None = None

        self.client = CopilotOAuthClient(
            self.config,
            self.storage,
            http_client,
            hook_manager=hook_manager,
            detection_service=detection_service,
        )

    @property
    def provider_name(self) -> str:
        """Internal provider name."""
        return "copilot"

    @property
    def provider_display_name(self) -> str:
        """Display name for UI."""
        return "GitHub Copilot"

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
        return False  # GitHub Device Code Flow doesn't require client secret

    async def get_authorization_url(
        self,
        state: str,
        code_verifier: str | None = None,
        redirect_uri: str | None = None,
    ) -> str:
        """Get the authorization URL for GitHub Device Code Flow.

        For device code flow, this returns the device authorization endpoint.
        The actual user verification happens at the verification_uri returned
        by start_device_flow().

        Args:
            state: OAuth state parameter (not used in device flow)
            code_verifier: PKCE code verifier (not used in device flow)

        Returns:
            Device authorization URL
        """
        # For device code flow, we return the device authorization endpoint
        # The actual flow is handled by the device flow methods
        return self.config.authorize_url

    async def start_device_flow(self) -> tuple[str, str, str, int]:
        """Start the GitHub device code authorization flow.

        Returns:
            Tuple of (device_code, user_code, verification_uri, expires_in)
        """
        device_response = await self.client.start_device_flow()

        logger.info(
            "device_flow_started",
            user_code=device_response.user_code,
            verification_uri=device_response.verification_uri,
            expires_in=device_response.expires_in,
        )

        return (
            device_response.device_code,
            device_response.user_code,
            device_response.verification_uri,
            device_response.expires_in,
        )

    async def complete_device_flow(
        self, device_code: str, interval: int = 5, expires_in: int = 900
    ) -> CopilotCredentials:
        """Complete the device flow authorization.

        Args:
            device_code: Device code from start_device_flow
            interval: Polling interval in seconds
            expires_in: Code expiration time in seconds

        Returns:
            Complete Copilot credentials
        """
        return await self.client.complete_authorization(
            device_code, interval, expires_in
        )

    async def handle_callback(
        self,
        code: str,
        state: str,
        code_verifier: str | None = None,
        redirect_uri: str | None = None,
    ) -> Any:
        """Handle OAuth callback (not used in device flow).

        This method is required by the CLI flow protocol but not used for
        device code flow. Use complete_device_flow instead.

        Args:
            code: Authorization code from OAuth callback
            state: State parameter for validation
            code_verifier: PKCE code verifier (if PKCE is used)
            redirect_uri: Redirect URI used in authorization (optional)
        """
        raise NotImplementedError(
            "Copilot uses device code flow. Browser callback is not supported."
        )

    async def exchange_code(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> dict[str, Any]:
        """Exchange authorization code for token (not used in device flow).

        This method is required by the OAuth protocol but not used for
        device code flow. Use complete_device_flow instead.
        """
        raise NotImplementedError(
            "Device code flow doesn't use authorization code exchange. "
            "Use complete_device_flow instead."
        )

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh access token using refresh token.

        For Copilot, this refreshes the Copilot service token using the
        stored OAuth token.

        Args:
            refresh_token: Not used for Copilot (uses OAuth token instead)

        Returns:
            Token information
        """
        credentials = await self.storage.load_credentials()
        if not credentials:
            raise ValueError("No credentials found for refresh")

        refreshed_credentials = await self.client.refresh_copilot_token(credentials)

        # Return token info in standard format
        if refreshed_credentials.copilot_token is not None:
            return {
                "access_token": refreshed_credentials.copilot_token.token.get_secret_value(),
                "token_type": "bearer",
                "expires_at": refreshed_credentials.copilot_token.expires_at,
                "provider": self.provider_name,
            }
        else:
            raise ValueError("Failed to refresh Copilot token")

    async def get_user_profile(self, access_token: str) -> StandardProfileFields:
        """Get user profile information.

        Args:
            access_token: OAuth access token (not Copilot token)

        Returns:
            User profile information
        """
        credentials = await self.storage.load_credentials()
        if not credentials:
            raise ValueError("No credentials found")

        # Get the actual profile info from the client
        profile = await self.client.get_user_profile(credentials.oauth_token)

        # Convert to StandardProfileFields
        display_name = getattr(profile, "computed_display_name", None)
        return StandardProfileFields(
            account_id=profile.account_id,
            provider_type="copilot",
            email=profile.email or None,
            display_name=display_name,
        )

    async def get_token_info(self) -> CopilotTokenInfo | None:
        """Get current token information.

        Returns:
            Token information if available
        """
        credentials = await self.storage.load_credentials()
        if not credentials:
            return None

        oauth_expires_at = credentials.oauth_token.expires_at_datetime
        copilot_expires_at = None

        if credentials.copilot_token and credentials.copilot_token.expires_at:
            from datetime import datetime

            with contextlib.suppress(ValueError):
                expires_str = credentials.copilot_token.expires_at
                # Handle both Z suffix and timezone offset formats
                if expires_str.endswith("Z"):
                    expires_str = expires_str[:-1] + "+00:00"
                copilot_expires_at = datetime.fromisoformat(expires_str)

        # Get profile for additional info
        profile = None
        with contextlib.suppress(Exception):
            profile = await self.get_user_profile("")

        return CopilotTokenInfo(
            provider="copilot",
            oauth_expires_at=oauth_expires_at,
            copilot_expires_at=copilot_expires_at,
            account_type=credentials.account_type,
            copilot_access=False,  # TODO: Get from profile or credentials
        )

    async def is_authenticated(self) -> bool:
        """Check if user is authenticated with valid tokens.

        Returns:
            True if authenticated with valid tokens
        """
        credentials = await self.storage.load_credentials()
        if not credentials:
            return False

        # Check if OAuth token is expired
        if credentials.oauth_token.is_expired:
            return False

        # Check if we have a Copilot token
        return credentials.copilot_token is not None

    async def get_copilot_token(self) -> str | None:
        """Get current Copilot service token for API requests.

        Returns:
            Copilot token if available and valid
        """
        credentials = await self.storage.load_credentials()
        if not credentials or not credentials.copilot_token:
            return None

        return credentials.copilot_token.token.get_secret_value()

    async def ensure_copilot_token(self) -> str:
        """Ensure we have a valid Copilot token, refreshing if necessary.

        Returns:
            Valid Copilot token

        Raises:
            ValueError: If unable to get valid token
        """
        credentials = await self.storage.load_credentials()
        if not credentials:
            raise ValueError("No credentials found - authorization required")

        if credentials.oauth_token.is_expired:
            raise ValueError("OAuth token expired - re-authorization required")

        # If no Copilot token or potentially expired, refresh it
        if not credentials.copilot_token:
            logger.info("no_copilot_token_refreshing")
            credentials = await self.client.refresh_copilot_token(credentials)

        if not credentials.copilot_token:
            raise ValueError("Failed to obtain Copilot token")

        return credentials.copilot_token.token.get_secret_value()

    async def logout(self) -> None:
        """Clear stored credentials."""
        await self.storage.clear_credentials()

    def _extract_standard_profile(self, credentials: Any) -> StandardProfileFields:
        """Extract standardized profile fields from Copilot credentials."""
        from .models import CopilotCredentials, CopilotProfileInfo

        if isinstance(credentials, CopilotProfileInfo):
            return StandardProfileFields(
                account_id=credentials.account_id,
                provider_type="copilot",
                email=credentials.email,
                display_name=credentials.name or credentials.login,
            )
        elif isinstance(credentials, CopilotCredentials):
            # Fallback for when we only have credentials without profile
            return StandardProfileFields(
                account_id="unknown",
                provider_type="copilot",
                email=None,
                display_name="GitHub Copilot User",
            )
        else:
            return StandardProfileFields(
                account_id="unknown",
                provider_type="copilot",
                email=None,
                display_name="Unknown User",
            )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.client.close()
        except Exception as e:
            logger.error(
                "provider_cleanup_failed",
                error=str(e),
                exc_info=e,
            )

    # OAuthProviderInfo protocol implementation

    @property
    def cli(self) -> CliAuthConfig:
        """Get CLI authentication configuration for this provider."""
        return CliAuthConfig(
            preferred_flow=FlowType.device,
            callback_port=8080,
            callback_path="/callback",
            supports_manual_code=False,
            supports_device_flow=True,
            fixed_redirect_uri=None,
        )

    def get_provider_info(self) -> OAuthProviderInfo:
        """Get provider information for registry."""
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="GitHub Copilot OAuth authentication",
            supports_pkce=self.supports_pkce,
            scopes=["read:user", "copilot"],
            is_available=True,
            plugin_name="copilot",
        )

    async def exchange_manual_code(self, code: str) -> Any:
        """Exchange manual authorization code for tokens.

        Note: Copilot primarily uses device code flow, but this method
        is provided for completeness.

        Args:
            code: Authorization code from manual entry

        Returns:
            Copilot credentials object
        """
        # Copilot doesn't typically support manual code entry as it uses device flow
        # This is a placeholder implementation
        raise NotImplementedError(
            "Copilot uses device code flow. Manual code entry is not supported."
        )
