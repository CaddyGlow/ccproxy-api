"""OAuth client implementation for Anthropic OAuth flow."""

import secrets
import urllib.parse
from typing import Optional

import httpx

from ccproxy.auth.oauth.models import OAuthTokenRequest, OAuthTokenResponse
from ccproxy.services.credentials.config import OAuthConfig
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class OAuthClient:
    """OAuth client for handling Anthropic OAuth flows."""

    def __init__(self, config: OAuthConfig | None = None):
        """Initialize OAuth client.

        Args:
            config: OAuth configuration, uses default if not provided
        """
        self.config = config or OAuthConfig()

    def generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge pair.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate code verifier (43-128 characters, URL-safe)
        code_verifier = secrets.token_urlsafe(96)  # 128 base64url chars

        # For now, use plain method (Anthropic supports this)
        # In production, should use SHA256 method
        code_challenge = code_verifier

        return code_verifier, code_challenge

    def build_authorization_url(self, state: str, code_challenge: str) -> str:
        """Build authorization URL for OAuth flow.

        Args:
            state: State parameter for CSRF protection
            code_challenge: PKCE code challenge

        Returns:
            Authorization URL
        """
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "plain",  # Using plain for simplicity
        }

        query_string = urllib.parse.urlencode(params)
        return f"{self.config.authorize_url}?{query_string}"

    async def exchange_code_for_tokens(
        self,
        authorization_code: str,
        code_verifier: str,
    ) -> OAuthTokenResponse:
        """Exchange authorization code for access tokens.

        Args:
            authorization_code: Authorization code from callback
            code_verifier: PKCE code verifier

        Returns:
            Token response

        Raises:
            httpx.HTTPError: If token exchange fails
        """
        token_request = OAuthTokenRequest(
            code=authorization_code,
            redirect_uri=self.config.redirect_uri,
            client_id=self.config.client_id,
            code_verifier=code_verifier,
        )

        headers = {
            "Content-Type": "application/json",
            "anthropic-beta": self.config.beta_version,
            "User-Agent": self.config.user_agent,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                headers=headers,
                json=token_request.model_dump(),
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )
                response.raise_for_status()

            data = response.json()
            return OAuthTokenResponse.model_validate(data)

    async def refresh_access_token(self, refresh_token: str) -> OAuthTokenResponse:
        """Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New token response

        Raises:
            httpx.HTTPError: If token refresh fails
        """
        refresh_request = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
        }

        headers = {
            "Content-Type": "application/json",
            "anthropic-beta": self.config.beta_version,
            "User-Agent": self.config.user_agent,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                headers=headers,
                json=refresh_request,
                timeout=30.0,
            )

            if response.status_code != 200:
                logger.error(
                    f"Token refresh failed: {response.status_code} - {response.text}"
                )
                response.raise_for_status()

            data = response.json()
            return OAuthTokenResponse.model_validate(data)
