"""Codex/OpenAI OAuth client implementation."""

from datetime import UTC, datetime
from typing import Any

import jwt

from ccproxy.auth.exceptions import OAuthError
from ccproxy.auth.models import OpenAICredentials
from ccproxy.auth.oauth.base import BaseOAuthClient
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger
from plugins.oauth_codex.config import CodexOAuthConfig


logger = get_plugin_logger()


class CodexOAuthClient(BaseOAuthClient[OpenAICredentials]):
    """Codex/OpenAI OAuth implementation for the OAuth Codex plugin."""

    def __init__(
        self,
        config: CodexOAuthConfig,
        storage: TokenStorage[OpenAICredentials] | None = None,
    ):
        """Initialize Codex OAuth client.

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
        """Get OpenAI OAuth authorization endpoint.

        Returns:
            Full authorization endpoint URL
        """
        return self.oauth_config.authorize_url

    def _get_token_endpoint(self) -> str:
        """Get OpenAI OAuth token exchange endpoint.

        Returns:
            Full token endpoint URL
        """
        return self.oauth_config.token_url

    def get_custom_auth_params(self) -> dict[str, str]:
        """Get OpenAI-specific authorization parameters.

        Returns:
            Dictionary of custom parameters
        """
        return {
            "audience": self.oauth_config.audience,
        }

    def get_custom_headers(self) -> dict[str, str]:
        """Get OpenAI-specific HTTP headers.

        Returns:
            Dictionary of custom headers
        """
        return {
            "User-Agent": self.oauth_config.user_agent,
        }

    async def parse_token_response(self, data: dict[str, Any]) -> OpenAICredentials:
        """Parse OpenAI-specific token response.

        Args:
            data: Raw token response from OpenAI

        Returns:
            OpenAI credentials object

        Raises:
            OAuthError: If response parsing fails
        """
        try:
            # Extract access token
            access_token = data["access_token"]
            refresh_token = data.get("refresh_token", "")

            # Calculate expiration
            expires_in = data.get("expires_in", 3600)
            expires_at = datetime.now(UTC).replace(microsecond=0)
            expires_at = expires_at.timestamp() + expires_in

            # Extract user info from ID token if present
            user_info = {}
            if "id_token" in data:
                try:
                    # Decode without verification for user info extraction
                    # In production, you should verify the JWT signature
                    decoded = jwt.decode(
                        data["id_token"],
                        options={"verify_signature": False},
                    )
                    user_info = {
                        "sub": decoded.get("sub"),
                        "email": decoded.get("email"),
                        "name": decoded.get("name"),
                        "picture": decoded.get("picture"),
                    }
                    logger.debug(
                        "codex_oauth_id_token_decoded",
                        sub=user_info.get("sub"),
                        email=user_info.get("email"),
                        category="auth",
                    )
                except Exception as e:
                    logger.warning(
                        "codex_oauth_id_token_decode_error",
                        error=str(e),
                        exc_info=e,
                        category="auth",
                    )

            # Extract account ID from ID token or use a default
            account_id = user_info.get("sub", "unknown") if user_info else "unknown"

            # Create credentials
            credentials = OpenAICredentials(
                access_token=access_token,
                refresh_token=refresh_token or "",
                id_token=data.get("id_token"),
                expires_at=datetime.fromtimestamp(expires_at, tz=UTC),
                account_id=account_id,
                active=True,
            )

            logger.info(
                "codex_oauth_credentials_parsed",
                has_refresh_token=bool(refresh_token),
                expires_in=expires_in,
                has_id_token=bool(data.get("id_token")),
                account_id=account_id,
                category="auth",
            )

            return credentials

        except KeyError as e:
            logger.error(
                "codex_oauth_token_response_missing_field",
                missing_field=str(e),
                response_keys=list(data.keys()),
                category="auth",
            )
            raise OAuthError(f"Missing required field in token response: {e}") from e
        except Exception as e:
            logger.error(
                "codex_oauth_token_response_parse_error",
                error=str(e),
                error_type=type(e).__name__,
                category="auth",
            )
            raise OAuthError(f"Failed to parse OpenAI token response: {e}") from e

    async def refresh_token(self, refresh_token: str) -> OpenAICredentials:
        """Refresh OpenAI access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New OpenAI credentials

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
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_endpoint,
                    data=data,  # OpenAI uses form encoding
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()

                token_response = response.json()
                return await self.parse_token_response(token_response)

        except Exception as e:
            logger.error(
                "codex_oauth_token_refresh_failed",
                error=str(e),
                exc_info=False,
                category="auth",
            )
            raise OAuthError(f"Failed to refresh OpenAI token: {e}") from e
