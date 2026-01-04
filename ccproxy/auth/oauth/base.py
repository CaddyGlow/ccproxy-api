"""Base OAuth client with common PKCE flow implementation."""

import asyncio
import base64
import hashlib
import secrets
import urllib.parse
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from typing import Any, Generic, TypeVar

import httpx

from ccproxy.auth.exceptions import (
    OAuthError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_logger
from ccproxy.http.client import HTTPClientFactory


logger = get_logger(__name__)

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class BaseOAuthClient(ABC, Generic[CredentialsT]):
    """Abstract base class for OAuth PKCE flow implementations."""

    def __init__(
        self,
        client_id: str,
        redirect_uri: str,
        base_url: str,
        scopes: list[str],
        storage: TokenStorage[CredentialsT] | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        settings: Settings | None = None,
    ):
        """Initialize OAuth client with common parameters.

        Args:
            client_id: OAuth client ID
            redirect_uri: OAuth callback redirect URI
            base_url: OAuth provider base URL
            scopes: List of OAuth scopes to request
            storage: Optional token storage backend
            http_client: Optional HTTP client (for request tracing support)
            hook_manager: Optional hook manager for emitting events
            settings: Optional settings for HTTP client configuration
        """
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.base_url = base_url
        self.scopes = scopes
        self.storage = storage
        self.hook_manager = hook_manager

        # Always have an HTTP client
        if http_client:
            self.http_client = http_client
            self._owns_http_client = False  # Don't close provided client
            logger.debug(
                "oauth_client_using_provided_http_client",
                http_client_id=id(http_client),
                has_hooks=hasattr(http_client, "hook_manager")
                and http_client.hook_manager is not None,
                hook_manager_id=id(hook_manager) if hook_manager else None,
            )
        else:
            # Create client with hook support if hook_manager is provided
            self.http_client = HTTPClientFactory.create_client(
                settings=settings,
                timeout_connect=10.0,
                timeout_read=30.0,
                http2=True,
                hook_manager=hook_manager,  # Pass hook manager to client
            )
            self._owns_http_client = True  # We own it, close on cleanup
            logger.debug(
                "oauth_client_created_new_http_client",
                http_client_id=id(self.http_client),
                has_hooks=hasattr(self.http_client, "hook_manager")
                and self.http_client.hook_manager is not None,
                hook_manager_id=id(hook_manager) if hook_manager else None,
            )

        self._callback_server: asyncio.Task[None] | None = None
        self._auth_complete = asyncio.Event()
        self._auth_result: Any | None = None
        self._auth_error: str | None = None

    async def close(self) -> None:
        """Close resources if we own them."""
        if self._owns_http_client and self.http_client:
            await self.http_client.aclose()

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if (
            self._owns_http_client
            and self.http_client
            and not self.http_client.is_closed
        ):
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                loop.create_task(self.http_client.aclose())
            except RuntimeError:
                # No running event loop, can't clean up async resources
                pass

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge.

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate code verifier (43-128 characters, URL-safe)
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")
        )

        # Generate code challenge using SHA256
        challenge_bytes = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode().rstrip("=")

        logger.debug(
            "pkce_pair_generated",
            verifier_length=len(code_verifier),
            challenge_length=len(code_challenge),
            category="auth",
        )
        return code_verifier, code_challenge

    def _generate_state(self) -> str:
        """Generate secure random state parameter.

        Returns:
            URL-safe random state string
        """
        return secrets.token_urlsafe(32)

    def _build_auth_url(self, code_challenge: str, state: str) -> str:
        """Build OAuth authorization URL with PKCE parameters.

        Args:
            code_challenge: PKCE code challenge
            state: Random state parameter

        Returns:
            Complete authorization URL
        """
        params = self._get_auth_params(code_challenge, state)
        query_string = urllib.parse.urlencode(params)
        auth_endpoint = self._get_auth_endpoint()
        return f"{auth_endpoint}?{query_string}"

    def _get_auth_params(self, code_challenge: str, state: str) -> dict[str, str]:
        """Get authorization URL parameters.

        Args:
            code_challenge: PKCE code challenge
            state: Random state parameter

        Returns:
            Dictionary of URL parameters
        """
        base_params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        # Allow providers to add custom parameters
        custom_params = self.get_custom_auth_params()
        base_params.update(custom_params)

        return base_params

    async def _exchange_code_for_tokens(
        self, code: str, code_verifier: str, state: str | None = None
    ) -> dict[str, Any]:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from OAuth callback
            code_verifier: PKCE code verifier
            state: OAuth state parameter

        Returns:
            Token response dictionary from provider

        Raises:
            OAuthTokenRefreshError: If token exchange fails
        """
        token_endpoint = self._get_token_endpoint()
        token_data = self._get_token_exchange_data(code, code_verifier, state)
        headers = self._get_token_exchange_headers()

        try:
            logger.debug(
                "token_exchange_start",
                endpoint=token_endpoint,
                has_code=bool(code),
                has_verifier=bool(code_verifier),
                category="auth",
            )

            # No need for OAuth-specific hooks here - generic HTTP hooks will capture everything

            # Just use self.http_client - it always exists!
            response = await self.http_client.post(
                token_endpoint,
                data=token_data if not self._use_json_for_token_exchange() else None,
                json=token_data if self._use_json_for_token_exchange() else None,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            token_response = response.json()

            # No need for OAuth-specific hooks here - generic HTTP hooks will capture everything
            logger.debug(
                "token_exchange_success",
                has_access_token="access_token" in token_response,
                has_refresh_token="refresh_token" in token_response,
                expires_in=token_response.get("expires_in"),
            )

            from typing import cast

            return cast(dict[str, Any], token_response)

        except httpx.HTTPStatusError as e:
            error_detail = self._extract_error_detail(e.response)
            logger.error(
                "token_exchange_http_error",
                status_code=e.response.status_code,
                error_detail=error_detail,
                exc_info=e,
            )

            # No need for OAuth-specific hooks here - generic HTTP hooks will capture everything

            raise OAuthTokenRefreshError(
                f"Token exchange failed: {error_detail}"
            ) from e

        except httpx.TimeoutException as e:
            logger.error(
                "token_exchange_timeout", error=str(e), exc_info=e, category="auth"
            )
            raise OAuthTokenRefreshError("Token exchange timed out") from e

        except httpx.HTTPError as e:
            logger.error(
                "token_exchange_http_error",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            raise OAuthTokenRefreshError(
                f"HTTP error during token exchange: {e}"
            ) from e

        except Exception as e:
            logger.error("token_exchange_unexpected_error", error=str(e), exc_info=e)
            raise OAuthTokenRefreshError(
                f"Unexpected error during token exchange: {e}"
            ) from e

    def _get_token_exchange_data(
        self, code: str, code_verifier: str, state: str | None = None
    ) -> dict[str, str]:
        """Get token exchange request data.

        Note: RFC 6749 Section 4.1.3 specifies that the state parameter should
        NOT be included in token exchange requests. However, some providers
        (like Claude) have non-standard implementations that require it.

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier
            state: OAuth state parameter

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

        # RFC 6749 compliant: state parameter should be excluded
        # Override in provider-specific clients if needed (e.g., Claude)

        # Allow providers to add custom parameters
        custom_data = self.get_custom_token_params()
        base_data.update(custom_data)

        return base_data

    def _get_token_exchange_headers(self) -> dict[str, str]:
        """Get headers for token exchange request.

        Returns:
            Dictionary of HTTP headers
        """
        base_headers = {
            "Accept": "application/json",
        }

        # Use form encoding by default, unless provider uses JSON
        if not self._use_json_for_token_exchange():
            base_headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            base_headers["Content-Type"] = "application/json"

        # Allow providers to add custom headers
        custom_headers = self.get_custom_headers()
        base_headers.update(custom_headers)

        return base_headers

    def _extract_error_detail(self, response: httpx.Response) -> str:
        """Extract error detail from HTTP response.

        Args:
            response: HTTP response object

        Returns:
            Human-readable error detail
        """
        try:
            error_data = response.json()
            return str(
                error_data.get(
                    "error_description", error_data.get("error", str(response.text))
                )
            )
        except Exception:
            return response.text[:200] if len(response.text) > 200 else response.text

    def _calculate_expiration(self, expires_in: int | None) -> datetime:
        """Calculate token expiration timestamp.

        Args:
            expires_in: Seconds until token expires (None defaults to 1 hour)

        Returns:
            Expiration datetime in UTC
        """
        expires_in = expires_in or 3600  # Default to 1 hour
        return datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=expires_in)

    # ==================== Abstract Methods ====================

    @abstractmethod
    async def parse_token_response(self, data: dict[str, Any]) -> CredentialsT:
        """Parse provider-specific token response into credentials.

        Args:
            data: Raw token response from provider

        Returns:
            Provider-specific credentials object
        """
        pass

    @abstractmethod
    def _get_auth_endpoint(self) -> str:
        """Get OAuth authorization endpoint URL.

        Returns:
            Full authorization endpoint URL
        """
        pass

    @abstractmethod
    def _get_token_endpoint(self) -> str:
        """Get OAuth token exchange endpoint URL.

        Returns:
            Full token endpoint URL
        """
        pass

    # ==================== Optional Override Methods ====================

    def get_custom_auth_params(self) -> dict[str, str]:
        """Get provider-specific authorization parameters.

        Override this to add custom parameters to auth URL.

        Returns:
            Dictionary of custom parameters (empty by default)
        """
        return {}

    def get_custom_token_params(self) -> dict[str, str]:
        """Get provider-specific token exchange parameters.

        Override this to add custom parameters to token request.

        Returns:
            Dictionary of custom parameters (empty by default)
        """
        return {}

    def get_custom_headers(self) -> dict[str, str]:
        """Get provider-specific HTTP headers.

        Override this to add custom headers to requests.

        Returns:
            Dictionary of custom headers (empty by default)
        """
        return {}

    def _use_json_for_token_exchange(self) -> bool:
        """Whether to use JSON instead of form encoding for token exchange.

        Override this if provider requires JSON body.

        Returns:
            False by default (uses form encoding)
        """
        return False

    # ==================== Public Methods ====================

    async def authenticate(
        self, code_verifier: str | None = None, state: str | None = None
    ) -> tuple[str, str, str]:
        """Start OAuth authentication flow.

        Args:
            code_verifier: Optional pre-generated PKCE verifier
            state: Optional pre-generated state parameter

        Returns:
            Tuple of (auth_url, code_verifier, state)
        """
        # Generate PKCE parameters if not provided
        if not code_verifier:
            code_verifier, code_challenge = self._generate_pkce_pair()
        else:
            # Calculate challenge from provided verifier
            challenge_bytes = hashlib.sha256(code_verifier.encode()).digest()
            code_challenge = (
                base64.urlsafe_b64encode(challenge_bytes).decode().rstrip("=")
            )

        # Generate state if not provided
        if not state:
            state = self._generate_state()

        # Build authorization URL
        auth_url = self._build_auth_url(code_challenge, state)

        logger.info(
            "oauth_flow_started",
            provider=self.__class__.__name__,
            has_storage=bool(self.storage),
            scopes=self.scopes,
        )

        return auth_url, code_verifier, state

    async def handle_callback(
        self, code: str, state: str, code_verifier: str
    ) -> CredentialsT:
        """Handle OAuth callback and exchange code for tokens.

        Args:
            code: Authorization code from callback
            state: State parameter from callback
            code_verifier: PKCE code verifier

        Returns:
            Provider-specific credentials object

        Raises:
            OAuthError: If callback handling fails
        """
        try:
            # Exchange code for tokens
            token_response = await self._exchange_code_for_tokens(
                code, code_verifier, state
            )

            # Parse provider-specific response
            credentials: CredentialsT = await self.parse_token_response(token_response)

            # Save to storage if available
            if self.storage:
                success = await self.storage.save(credentials)
                if not success:
                    logger.warning(
                        "credentials_save_failed", provider=self.__class__.__name__
                    )

            logger.info(
                "oauth_callback_success",
                provider=self.__class__.__name__,
                has_refresh_token=bool(token_response.get("refresh_token")),
            )

            return credentials

        except OAuthTokenRefreshError:
            raise
        except Exception as e:
            logger.error(
                "oauth_callback_error",
                provider=self.__class__.__name__,
                error=str(e),
                exc_info=e,
            )
            raise OAuthError(f"OAuth callback failed: {e}") from e
