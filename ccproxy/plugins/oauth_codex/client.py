"""Codex/OpenAI OAuth client implementation."""

from datetime import UTC, datetime
from typing import Any

import httpx
import jwt
from pydantic import SecretStr

from ccproxy.auth.exceptions import OAuthError
from ccproxy.auth.oauth.base import BaseOAuthClient
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_plugin_logger

from .config import CodexOAuthConfig
from .models import OpenAICredentials, OpenAITokens


logger = get_plugin_logger()


class CodexOAuthClient(BaseOAuthClient[OpenAICredentials]):
    """Codex/OpenAI OAuth implementation for the OAuth Codex plugin."""

    def __init__(
        self,
        config: CodexOAuthConfig,
        storage: TokenStorage[OpenAICredentials] | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        settings: Settings | None = None,
    ):
        """Initialize Codex OAuth client.

        Args:
            config: OAuth configuration
            storage: Token storage backend
            http_client: Optional HTTP client (for request tracing support)
            hook_manager: Optional hook manager for emitting events
            settings: Optional settings for HTTP client configuration
        """
        self.oauth_config = config

        # Resolve effective redirect URI from config
        redirect_uri = config.get_redirect_uri()

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
        # OpenAI does not use the audience parameter in authorization requests
        return {}

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
            # Extract tokens
            access_token: str = data["access_token"]
            refresh_token: str = data.get("refresh_token", "")
            id_token: str = data.get("id_token", "")

            # Build credentials in the current nested schema; legacy inputs are also accepted
            # by the model's validator if needed.
            tokens = OpenAITokens(
                id_token=SecretStr(id_token),
                access_token=SecretStr(access_token),
                refresh_token=SecretStr(refresh_token or ""),
                account_id="",
            )
            credentials = OpenAICredentials(
                OPENAI_API_KEY=None,
                tokens=tokens,
                last_refresh=datetime.now(UTC).replace(microsecond=0).isoformat(),
                active=True,
            )

            # Try to extract account_id from JWT claims (id_token preferred)
            try:
                token_to_decode = id_token or access_token
                decoded = jwt.decode(
                    token_to_decode, options={"verify_signature": False}
                )
                account_id = (
                    decoded.get("sub")
                    or decoded.get("account_id")
                    or decoded.get("org_id")
                    or ""
                )
                # Pydantic model has properties mapping; update underlying field
                credentials.tokens.account_id = str(account_id)
                logger.debug(
                    "codex_oauth_id_token_decoded",
                    sub=decoded.get("sub"),
                    email=decoded.get("email"),
                    category="auth",
                )
            except Exception as e:
                logger.warning(
                    "codex_oauth_id_token_decode_error",
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )

            logger.info(
                "codex_oauth_credentials_parsed",
                has_refresh_token=bool(refresh_token),
                has_id_token=bool(id_token),
                account_id=credentials.account_id,
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
            "scope": "openid profile email offline_access",
        }
        headers = self.get_custom_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"

        try:
            response = await self.http_client.post(
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
