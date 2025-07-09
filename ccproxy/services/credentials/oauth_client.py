"""OAuth client for Claude authentication."""

import base64
import hashlib
import os
import secrets
import urllib.parse
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Union
from urllib.parse import parse_qs, urlparse

import httpx

from ccproxy.services.credentials.config import OAuthConfig
from ccproxy.services.credentials.exceptions import (
    OAuthCallbackError,
    OAuthLoginError,
    OAuthTokenRefreshError,
)
from ccproxy.services.credentials.models import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class OAuthClient:
    """Client for handling OAuth authentication flow."""

    def __init__(
        self,
        config: OAuthConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize OAuth client.

        Args:
            config: OAuth configuration (uses defaults if not provided)
            http_client: HTTP client for making requests (creates one if not provided)
        """
        self.config = config or OAuthConfig()
        self._http_client = http_client
        self._owns_http_client = http_client is None

    def _get_proxy_url(self) -> str | None:
        """Get proxy URL from environment variables.

        Returns:
            str or None: Proxy URL if any proxy is set
        """
        # Check for standard proxy environment variables
        # For HTTPS requests, prioritize HTTPS_PROXY
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        all_proxy = os.environ.get("ALL_PROXY")
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

        proxy_url = https_proxy or all_proxy or http_proxy

        if proxy_url:
            logger.debug(f"Using proxy: {proxy_url}")

        return proxy_url

    def _get_ssl_context(self) -> str | bool:
        """Get SSL context configuration from environment variables.

        Returns:
            SSL verification configuration:
            - Path to CA bundle file
            - True for default verification
            - False to disable verification (insecure)
        """
        # Check for custom CA bundle
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get(
            "SSL_CERT_FILE"
        )

        # Check if SSL verification should be disabled (NOT RECOMMENDED)
        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower()

        if ca_bundle and Path(ca_bundle).exists():
            logger.debug(f"Using custom CA bundle: {ca_bundle}")
            return ca_bundle
        elif ssl_verify in ("false", "0", "no"):
            logger.warning("SSL verification disabled - this is insecure!")
            return False
        else:
            return True

    async def __aenter__(self) -> "OAuthClient":
        """Async context manager entry."""
        if self._http_client is None:
            proxy_url = self._get_proxy_url()
            verify = self._get_ssl_context()
            self._http_client = httpx.AsyncClient(proxy=proxy_url, verify=verify)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owns_http_client and self._http_client:
            await self._http_client.aclose()

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Get the HTTP client, creating one if needed."""
        if self._http_client is None:
            proxy_url = self._get_proxy_url()
            verify = self._get_ssl_context()
            self._http_client = httpx.AsyncClient(proxy=proxy_url, verify=verify)
        return self._http_client

    async def login(self) -> ClaudeCredentials:
        """Perform OAuth login flow.

        Returns:
            ClaudeCredentials with OAuth token

        Raises:
            OAuthLoginError: If login fails
            OAuthCallbackError: If callback processing fails
        """
        # Generate state parameter for security
        state = secrets.token_urlsafe(32)

        # Generate PKCE parameters
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )

        authorization_code = None
        error = None

        class OAuthCallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                nonlocal authorization_code, error

                # Ignore favicon requests
                if self.path == "/favicon.ico":
                    self.send_response(404)
                    self.end_headers()
                    return

                parsed_url = urlparse(self.path)
                query_params = parse_qs(parsed_url.query)

                # Check state parameter
                received_state = query_params.get("state", [None])[0]

                if received_state != state:
                    error = "Invalid state parameter"
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error: Invalid state parameter")
                    return

                # Check for authorization code
                if "code" in query_params:
                    authorization_code = query_params["code"][0]
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Login successful! You can close this window.")
                elif "error" in query_params:
                    error = query_params.get("error_description", ["Unknown error"])[0]
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(f"Error: {error}".encode())
                else:
                    error = "No authorization code received"
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error: No authorization code received")

            def log_message(self, format: str, *args: Any) -> None:
                # Suppress HTTP server logs
                pass

        # Start local HTTP server for OAuth callback
        server = HTTPServer(
            ("localhost", self.config.callback_port), OAuthCallbackHandler
        )
        server_thread = Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        try:
            # Build authorization URL
            auth_params = {
                "response_type": "code",
                "client_id": self.config.client_id,
                "redirect_uri": self.config.redirect_uri,
                "scope": " ".join(self.config.scopes),
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }

            auth_url = (
                f"{self.config.authorize_url}?{urllib.parse.urlencode(auth_params)}"
            )

            logger.info("Opening browser for OAuth authorization...")
            logger.info(f"If browser doesn't open, visit: {auth_url}")

            # Open browser
            webbrowser.open(auth_url)

            # Wait for callback (with timeout)
            import time

            start_time = time.time()

            while authorization_code is None and error is None:
                if time.time() - start_time > self.config.callback_timeout:
                    error = "Login timeout"
                    break
                time.sleep(0.1)

            if error:
                raise OAuthCallbackError(f"OAuth callback failed: {error}")

            if not authorization_code:
                raise OAuthLoginError("No authorization code received")

            # Exchange authorization code for tokens
            token_data = {
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
                "code_verifier": code_verifier,
                "state": state,
            }

            headers = {
                "Content-Type": "application/json",
                "anthropic-beta": self.config.beta_version,
                "User-Agent": self.config.user_agent,
            }

            response = await self.http_client.post(
                self.config.token_url,
                headers=headers,
                json=token_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()

                # Calculate expires_at from expires_in
                expires_in = result.get("expires_in")
                expires_at = None
                if expires_in:
                    expires_at = int(
                        (datetime.now(UTC).timestamp() + expires_in) * 1000
                    )

                # Create credentials object
                oauth_data = {
                    "accessToken": result.get("access_token"),
                    "refreshToken": result.get("refresh_token"),
                    "expiresAt": expires_at,
                    "scopes": result.get("scope", "").split()
                    if result.get("scope")
                    else self.config.scopes,
                    "subscriptionType": result.get("subscription_type", "unknown"),
                }

                credentials = ClaudeCredentials(claudeAiOauth=OAuthToken(**oauth_data))

                logger.info("Successfully completed OAuth login")
                return credentials

            else:
                raise OAuthLoginError(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )

        except Exception as e:
            if isinstance(e, OAuthLoginError | OAuthCallbackError):
                raise
            raise OAuthLoginError(f"OAuth login failed: {e}") from e

        finally:
            # Stop the HTTP server
            server.shutdown()
            server_thread.join(timeout=1)

    async def refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh an OAuth access token.

        Args:
            refresh_token: The refresh token to use

        Returns:
            New OAuth token with updated access token

        Raises:
            OAuthTokenRefreshError: If token refresh fails
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "anthropic-beta": self.config.beta_version,
                "User-Agent": self.config.user_agent,
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.config.client_id,
            }

            response = await self.http_client.post(
                self.config.token_url,
                headers=headers,
                json=data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()

                # Calculate expires_at from expires_in (seconds)
                expires_in = result.get("expires_in")
                expires_at = None
                if expires_in:
                    expires_at = int(
                        (datetime.now(UTC).timestamp() + expires_in) * 1000
                    )  # Convert to milliseconds

                oauth_data = {
                    "accessToken": result.get("access_token"),
                    "refreshToken": result.get("refresh_token", refresh_token),
                    "expiresAt": expires_at,
                    "scopes": result.get("scope", "").split()
                    if result.get("scope")
                    else [],
                    "subscriptionType": "unknown",  # Not returned in refresh
                }

                logger.debug("Successfully refreshed OAuth token")
                return OAuthToken(**oauth_data)

            else:
                raise OAuthTokenRefreshError(
                    f"Failed to refresh token: {response.status_code} - {response.text}"
                )

        except httpx.RequestError as e:
            raise OAuthTokenRefreshError(
                f"Network error during token refresh: {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OAuthTokenRefreshError):
                raise
            raise OAuthTokenRefreshError(f"Token refresh failed: {e}") from e

    async def fetch_user_profile(self, access_token: str) -> UserProfile:
        """Fetch user profile using OAuth token.

        Uses the correct profile API endpoint with the access token.

        Args:
            access_token: Current access token to use for authentication
            refresh_token: Refresh token (not used, kept for compatibility)

        Returns:
            UserProfile with organization and account info

        Raises:
            OAuthTokenRefreshError: If the request fails
        """
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "anthropic-beta": self.config.beta_version,
                "User-Agent": self.config.user_agent,
            }

            response = await self.http_client.get(
                "https://api.anthropic.com/api/oauth/profile",
                headers=headers,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()

                # Extract organization and account info
                profile = UserProfile(
                    organization=OrganizationInfo(**result.get("organization", {}))
                    if result.get("organization")
                    else None,
                    account=AccountInfo(**result.get("account", {}))
                    if result.get("account")
                    else None,
                )
                return profile
            else:
                raise OAuthTokenRefreshError(
                    f"Failed to fetch user profile: {response.status_code} - {response.text}"
                )

        except Exception as e:
            if isinstance(e, OAuthTokenRefreshError):
                raise
            raise OAuthTokenRefreshError(f"Error fetching user profile: {e}") from e
