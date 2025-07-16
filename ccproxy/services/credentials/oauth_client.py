"""OAuth client implementation for Anthropic OAuth flow."""

import asyncio
import base64
import hashlib
import secrets
import time
import urllib.parse
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from structlog import get_logger

from ccproxy.auth.exceptions import OAuthCallbackError, OAuthLoginError
from ccproxy.auth.models import ClaudeCredentials, OAuthToken, UserProfile
from ccproxy.auth.oauth.models import OAuthTokenRequest, OAuthTokenResponse
from ccproxy.config.auth import OAuthSettings
from ccproxy.services.credentials.config import OAuthConfig


logger = get_logger(__name__)


def _log_http_error_compact(operation: str, response: httpx.Response) -> None:
    """Log HTTP error response in compact format.

    Args:
        operation: Description of the operation that failed
        response: HTTP response object
    """
    import os

    # Check if verbose API logging is enabled
    verbose_api = os.environ.get("CCPROXY_VERBOSE_API", "false").lower() == "true"

    if verbose_api:
        # Full verbose logging
        logger.error(
            "http_operation_failed",
            operation=operation,
            status_code=response.status_code,
            response_text=response.text,
        )
    else:
        # Compact logging - truncate response body
        response_text = response.text
        if len(response_text) > 200:
            response_preview = f"{response_text[:100]}...{response_text[-50:]}"
        elif len(response_text) > 100:
            response_preview = f"{response_text[:100]}..."
        else:
            response_preview = response_text

        logger.error(
            "http_operation_failed_compact",
            operation=operation,
            status_code=response.status_code,
            response_preview=response_preview,
            verbose_hint="use CCPROXY_VERBOSE_API=true for full response",
        )


class OAuthClient:
    """OAuth client for handling Anthropic OAuth flows."""

    def __init__(self, config: OAuthSettings | None = None):
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
                timeout=self.config.request_timeout,
            )

            if response.status_code != 200:
                _log_http_error_compact("Token exchange", response)
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
                timeout=self.config.request_timeout,
            )

            if response.status_code != 200:
                _log_http_error_compact("Token refresh", response)
                response.raise_for_status()

            data = response.json()
            return OAuthTokenResponse.model_validate(data)

    async def refresh_token(self, refresh_token: str) -> "OAuthToken":
        """Refresh token using refresh token - compatibility method for tests.

        Args:
            refresh_token: Refresh token

        Returns:
            New OAuth token

        Raises:
            OAuthTokenRefreshError: If token refresh fails
        """
        from datetime import UTC, datetime

        from ccproxy.auth.exceptions import OAuthTokenRefreshError
        from ccproxy.auth.models import OAuthToken

        try:
            token_response = await self.refresh_access_token(refresh_token)

            expires_in = (
                token_response.expires_in if token_response.expires_in else 3600
            )

            # Convert to OAuthToken format expected by tests
            expires_at_ms = int((datetime.now(UTC).timestamp() + expires_in) * 1000)

            return OAuthToken(
                accessToken=token_response.access_token,
                refreshToken=token_response.refresh_token or refresh_token,
                expiresAt=expires_at_ms,
                scopes=token_response.scope.split() if token_response.scope else [],
                subscriptionType="pro",  # Default value
            )
        except Exception as e:
            raise OAuthTokenRefreshError(f"Token refresh failed: {e}") from e

    async def fetch_user_profile(self, access_token: str) -> UserProfile | None:
        """Fetch user profile information using access token.

        Args:
            access_token: Valid OAuth access token

        Returns:
            User profile information

        Raises:
            httpx.HTTPError: If profile fetch fails
        """
        from ccproxy.auth.models import UserProfile

        headers = {
            "Authorization": f"Bearer {access_token}",
            "anthropic-beta": self.config.beta_version,
            "User-Agent": self.config.user_agent,
            "Content-Type": "application/json",
        }

        # Use the profile url
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.config.profile_url,
                headers=headers,
                timeout=self.config.request_timeout,
            )

            if response.status_code == 404:
                # Userinfo endpoint not available - this is expected for some OAuth providers
                logger.debug(
                    "userinfo_endpoint_unavailable", endpoint=self.config.profile_url
                )
                return None
            elif response.status_code != 200:
                _log_http_error_compact("Profile fetch", response)
                response.raise_for_status()

            data = response.json()
            logger.debug("user_profile_fetched", endpoint=self.config.profile_url)
            return UserProfile.model_validate(data)

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

            logger.info("oauth_browser_opening", auth_url=auth_url)
            logger.info(
                "oauth_manual_url",
                message="If browser doesn't open, visit this URL",
                auth_url=auth_url,
            )

            # Open browser
            webbrowser.open(auth_url)

            # Wait for callback (with timeout)
            import time

            start_time = time.time()

            while authorization_code is None and error is None:
                if time.time() - start_time > self.config.callback_timeout:
                    error = "Login timeout"
                    break
                await asyncio.sleep(0.1)

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

            async with httpx.AsyncClient() as client:
                response = await client.post(
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

                logger.info("oauth_login_completed", client_id=self.config.client_id)
                return credentials

            else:
                # Use compact logging for the error message
                import os

                verbose_api = (
                    os.environ.get("CCPROXY_VERBOSE_API", "false").lower() == "true"
                )

                if verbose_api:
                    error_detail = response.text
                else:
                    response_text = response.text
                    if len(response_text) > 200:
                        error_detail = f"{response_text[:100]}...{response_text[-50:]}"
                    elif len(response_text) > 100:
                        error_detail = f"{response_text[:100]}..."
                    else:
                        error_detail = response_text

                raise OAuthLoginError(
                    f"Token exchange failed: {response.status_code} - {error_detail}"
                )

        except Exception as e:
            if isinstance(e, OAuthLoginError | OAuthCallbackError):
                raise
            raise OAuthLoginError(f"OAuth login failed: {e}") from e

        finally:
            # Stop the HTTP server
            server.shutdown()
            server_thread.join(timeout=1)
