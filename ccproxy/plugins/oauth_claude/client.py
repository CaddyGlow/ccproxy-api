"""Claude OAuth client implementation."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ccproxy.services.cli_detection import CLIDetectionService

import httpx
from pydantic import SecretStr

from ccproxy.auth.exceptions import OAuthError
from ccproxy.auth.oauth.base import BaseOAuthClient
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger

from .config import ClaudeOAuthConfig
from .models import (
    ClaudeCredentials,
    ClaudeOAuthToken,
)


logger = get_plugin_logger()


class ClaudeOAuthClient(BaseOAuthClient[ClaudeCredentials]):
    """Claude OAuth implementation for the OAuth Claude plugin."""

    def __init__(
        self,
        config: ClaudeOAuthConfig,
        storage: TokenStorage[ClaudeCredentials] | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: "CLIDetectionService | None" = None,
        settings: Settings | None = None,
    ):
        """Initialize Claude OAuth client.

        Args:
            config: OAuth configuration
            storage: Token storage backend
            http_client: Optional HTTP client (for request tracing support)
            hook_manager: Optional hook manager for emitting events
            detection_service: Optional CLI detection service for headers
            settings: Optional settings for HTTP client configuration
        """
        self.oauth_config = config
        self.detection_service = detection_service

        # Resolve effective redirect URI from config
        redirect_uri = config.get_redirect_uri()

        # Debug logging for CLI tracing
        logger.debug(
            "claude_oauth_client_init",
            has_http_client=http_client is not None,
            has_hook_manager=hook_manager is not None,
            http_client_id=id(http_client) if http_client else None,
            hook_manager_id=id(hook_manager) if hook_manager else None,
        )

        # Initialize base class
        super().__init__(
            client_id=config.client_id,
            redirect_uri=redirect_uri,
            base_url=config.base_url,
            scopes=config.scopes,
            storage=storage,
            http_client=http_client,
            hook_manager=hook_manager,
            settings=settings,
        )

    def _get_auth_endpoint(self) -> str:
        """Get Claude OAuth authorization endpoint.

        Returns:
            Full authorization endpoint URL
        """
        return self.oauth_config.authorize_url

    def _get_token_endpoint(self) -> str:
        """Get Claude OAuth token exchange endpoint.

        Returns:
            Full token endpoint URL
        """
        return self.oauth_config.token_url

    def get_custom_headers(self) -> dict[str, str]:
        """Get Claude-specific HTTP headers.

        Returns:
            Dictionary of custom headers
        """
        # Start with headers from config
        headers = dict(self.oauth_config.headers)

        # Use injected detection service if available
        if self.detection_service:
            try:
                get_headers = getattr(
                    self.detection_service, "get_cached_headers", None
                )
                detected_headers = get_headers() if callable(get_headers) else None
                if detected_headers and "user-agent" in detected_headers:
                    headers["User-Agent"] = detected_headers["user-agent"]
            except Exception:
                # Keep the User-Agent from config if detection service not available
                pass
        # No fallback - if detection service is not injected, use config headers only

        return headers

    def _use_json_for_token_exchange(self) -> bool:
        """Claude uses JSON for token exchange.

        Returns:
            True to use JSON body
        """
        return True

    def _get_token_exchange_data(
        self, code: str, code_verifier: str, state: str | None = None
    ) -> dict[str, str]:
        """Get token exchange request data for Claude.

        Claude has a non-standard OAuth implementation that requires the
        state parameter in token exchange requests, unlike RFC 6749 Section 4.1.3.

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier
            state: OAuth state parameter (required by Claude)

        Returns:
            Dictionary of token exchange parameters
        """
        base_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }

        # Claude requires the state parameter in token exchange (non-standard)
        if state:
            base_data["state"] = state

        # Allow for custom parameters
        custom_data = self.get_custom_token_params()
        base_data.update(custom_data)

        return base_data

    async def parse_token_response(self, data: dict[str, Any]) -> ClaudeCredentials:
        """Parse Claude-specific token response.

        Args:
            data: Raw token response from Claude

        Returns:
            Claude credentials object

        Raises:
            OAuthError: If response parsing fails
        """
        try:
            # Calculate expiration time
            expires_in = data.get("expires_in")
            expires_at = None
            if expires_in:
                expires_at = int((datetime.now(UTC).timestamp() + expires_in) * 1000)

            # Parse scope string into list
            scopes: list[str] = []
            if data.get("scope"):
                scopes = (
                    data["scope"].split()
                    if isinstance(data["scope"], str)
                    else data["scope"]
                )

            # Create OAuth token
            oauth_token = ClaudeOAuthToken(
                accessToken=SecretStr(data["access_token"]),
                refreshToken=SecretStr(data.get("refresh_token", "")),
                expiresAt=expires_at,
                scopes=scopes or self.oauth_config.scopes,
                subscriptionType=data.get("subscription_type"),
            )

            # Create credentials (using alias for field name)
            credentials = ClaudeCredentials(claudeAiOauth=oauth_token)

            logger.info(
                "claude_oauth_credentials_parsed",
                has_refresh_token=bool(data.get("refresh_token")),
                expires_in=expires_in,
                subscription_type=oauth_token.subscription_type,
                scopes=oauth_token.scopes,
                category="auth",
            )

            return credentials

        except KeyError as e:
            logger.error(
                "claude_oauth_token_response_missing_field",
                missing_field=str(e),
                response_keys=list(data.keys()),
                category="auth",
            )
            raise OAuthError(f"Missing required field in token response: {e}") from e
        except Exception as e:
            logger.error(
                "claude_oauth_token_response_parse_error",
                error=str(e),
                error_type=type(e).__name__,
                category="auth",
            )
            raise OAuthError(f"Failed to parse Claude token response: {e}") from e

    async def refresh_token(self, refresh_token: str) -> ClaudeCredentials:
        """Refresh Claude access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New Claude credentials

        Raises:
            OAuthError: If refresh fails
        """
        token_endpoint = self._get_token_endpoint()
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }
        headers = self.get_custom_headers()
        headers["Content-Type"] = "application/json"

        try:
            # Use the HTTP client directly (always available now)
            response = await self.http_client.post(
                token_endpoint,
                json=data,  # Claude uses JSON
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            token_response = response.json()
            return await self.parse_token_response(token_response)

        except Exception as e:
            logger.error(
                "claude_oauth_token_refresh_failed",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            raise OAuthError(f"Failed to refresh Claude token: {e}") from e
