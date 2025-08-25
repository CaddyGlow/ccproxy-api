"""Claude OAuth client implementation."""

from datetime import UTC, datetime
from typing import Any

from pydantic import SecretStr

from ccproxy.auth.exceptions import OAuthError
from ccproxy.auth.models import ClaudeCredentials, OAuthToken
from ccproxy.auth.oauth.base import BaseOAuthClient
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger
from plugins.oauth_claude.config import ClaudeOAuthConfig


logger = get_plugin_logger()


class ClaudeOAuthClient(BaseOAuthClient[ClaudeCredentials]):
    """Claude OAuth implementation for the OAuth Claude plugin."""

    def __init__(
        self,
        config: ClaudeOAuthConfig,
        storage: TokenStorage[ClaudeCredentials] | None = None,
    ):
        """Initialize Claude OAuth client.

        Args:
            config: OAuth configuration
            storage: Token storage backend
        """
        self.oauth_config = config

        # Initialize base class
        super().__init__(
            client_id=config.client_id,
            redirect_uri=config.redirect_uri,
            base_url=config.base_url,
            scopes=config.scopes,
            storage=storage,
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
        return {
            "anthropic-beta": self.oauth_config.beta_version,
            "User-Agent": self.oauth_config.user_agent,
        }

    def _use_json_for_token_exchange(self) -> bool:
        """Claude uses JSON for token exchange.

        Returns:
            True to use JSON body
        """
        return True

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
            oauth_token = OAuthToken(
                accessToken=SecretStr(data["access_token"]),
                refreshToken=SecretStr(data.get("refresh_token", "")),
                expiresAt=expires_at,
                scopes=scopes or self.oauth_config.scopes,
                subscriptionType=data.get("subscription_type", "unknown"),
                tokenType=data.get("token_type", "Bearer"),
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
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
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

    def _extract_subscription_info(self, token_data: dict[str, Any]) -> str:
        """Extract subscription type from token response.

        Args:
            token_data: Token response data

        Returns:
            Subscription type string
        """
        # Check for subscription_type in response
        if "subscription_type" in token_data:
            return str(token_data["subscription_type"])

        # Check for plan information in scope
        scope = token_data.get("scope", "")
        if isinstance(scope, list):
            scope = " ".join(scope)

        if "claude-pro" in scope.lower():
            return "claude-pro"
        elif "claude-max" in scope.lower():
            return "claude-max"
        elif "free" in scope.lower():
            return "free"

        return "unknown"
