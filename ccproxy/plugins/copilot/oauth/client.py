"""OAuth client implementation for GitHub Copilot with Device Code Flow."""

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr

from ccproxy.core.logging import get_plugin_logger

from ..config import CopilotOAuthConfig
from .models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotProfileInfo,
    CopilotTokenResponse,
    DeviceCodeResponse,
    DeviceTokenPollResponse,
)
from .storage import CopilotOAuthStorage


if TYPE_CHECKING:
    from ccproxy.services.cli_detection import CLIDetectionService


logger = get_plugin_logger()


class CopilotOAuthClient:
    """OAuth client for GitHub Copilot using Device Code Flow."""

    def __init__(
        self,
        config: CopilotOAuthConfig,
        storage: CopilotOAuthStorage,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: "CLIDetectionService | None" = None,
    ):
        """Initialize the OAuth client.

        Args:
            config: OAuth configuration
            storage: Token storage
            http_client: Optional HTTP client for request tracing
            hook_manager: Optional hook manager for events
            detection_service: Optional CLI detection service
        """
        self.config = config
        self.storage = storage
        self.hook_manager = hook_manager
        self.detection_service = detection_service
        self._http_client = http_client
        self._owns_client = http_client is None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get HTTP client for making requests."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "CCProxy-Copilot/1.0.0",
                },
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client if we own it."""
        if self._owns_client and self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def start_device_flow(self) -> DeviceCodeResponse:
        """Start the GitHub device code authorization flow.

        Returns:
            Device code response with verification details
        """
        client = await self._get_http_client()

        # Request device code from GitHub
        data = {
            "client_id": self.config.client_id,
            "scope": " ".join(self.config.scopes),
        }

        logger.debug(
            "requesting_device_code",
            client_id=self.config.client_id[:8] + "...",
            scopes=self.config.scopes,
        )

        try:
            response = await client.post(
                self.config.authorize_url,
                data=data,
                headers={
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()

            device_code_data = response.json()
            device_code_response = DeviceCodeResponse.model_validate(device_code_data)

            logger.info(
                "device_code_received",
                user_code=device_code_response.user_code,
                verification_uri=device_code_response.verification_uri,
                expires_in=device_code_response.expires_in,
            )

            return device_code_response

        except httpx.HTTPError as e:
            logger.error(
                "device_code_request_failed",
                error=str(e),
                status_code=getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                exc_info=e,
            )
            raise

    async def poll_for_token(
        self, device_code: str, interval: int, expires_in: int
    ) -> CopilotOAuthToken:
        """Poll GitHub for OAuth token after user authorization.

        Args:
            device_code: Device code from device flow
            interval: Polling interval in seconds
            expires_in: Code expiration time in seconds

        Returns:
            OAuth token once authorized

        Raises:
            TimeoutError: If device code expires
            ValueError: If user denies authorization
        """
        client = await self._get_http_client()

        start_time = time.time()
        current_interval = interval

        logger.info(
            "polling_for_token",
            interval=interval,
            expires_in=expires_in,
        )

        while True:
            # Check if we've exceeded the expiration time
            if time.time() - start_time > expires_in:
                raise TimeoutError("Device code has expired")

            await asyncio.sleep(current_interval)

            data = {
                "client_id": self.config.client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            }

            try:
                response = await client.post(
                    self.config.token_url,
                    data=data,
                    headers={
                        "Accept": "application/json",
                    },
                )

                poll_response = DeviceTokenPollResponse.model_validate(response.json())

                if poll_response.is_success:
                    # Success! Create OAuth token
                    oauth_token = CopilotOAuthToken(
                        access_token=SecretStr(poll_response.access_token or ""),
                        token_type=poll_response.token_type or "bearer",
                        scope=poll_response.scope or " ".join(self.config.scopes),
                        created_at=int(time.time()),
                        expires_in=None,  # GitHub tokens don't typically expire
                    )

                    logger.info(
                        "oauth_token_received",
                        token_type=oauth_token.token_type,
                        scope=oauth_token.scope,
                    )

                    return oauth_token

                elif poll_response.is_pending:
                    # Still waiting for user authorization
                    logger.debug("authorization_pending")
                    continue

                elif poll_response.is_slow_down:
                    # Need to slow down polling
                    current_interval += 5
                    logger.debug("slowing_down_poll", new_interval=current_interval)
                    continue

                elif poll_response.is_expired:
                    raise TimeoutError("Device code has expired")

                elif poll_response.is_denied:
                    raise ValueError("User denied authorization")

                else:
                    # Unknown error
                    logger.error(
                        "unknown_oauth_error",
                        error=poll_response.error,
                        error_description=poll_response.error_description,
                    )
                    raise ValueError(f"OAuth error: {poll_response.error}")

            except httpx.HTTPError as e:
                logger.error(
                    "token_poll_request_failed",
                    error=str(e),
                    status_code=getattr(e.response, "status_code", None)
                    if hasattr(e, "response")
                    else None,
                    exc_info=e,
                )
                # Continue polling on HTTP errors
                await asyncio.sleep(current_interval)
                continue

    async def exchange_for_copilot_token(
        self, oauth_token: CopilotOAuthToken
    ) -> CopilotTokenResponse:
        """Exchange GitHub OAuth token for Copilot service token.

        Args:
            oauth_token: GitHub OAuth token

        Returns:
            Copilot service token response
        """
        client = await self._get_http_client()

        logger.debug(
            "exchanging_for_copilot_token",
            copilot_token_url=self.config.copilot_token_url,
        )

        try:
            response = await client.get(
                self.config.copilot_token_url,
                headers={
                    "Authorization": f"Bearer {oauth_token.access_token.get_secret_value()}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()

            copilot_data = response.json()
            copilot_token = CopilotTokenResponse.model_validate(copilot_data)

            logger.info(
                "copilot_token_received",
                expires_at=copilot_token.expires_at,
                refresh_in=copilot_token.refresh_in,
            )

            return copilot_token

        except httpx.HTTPError as e:
            logger.error(
                "copilot_token_exchange_failed",
                error=str(e),
                status_code=getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                exc_info=e,
            )
            raise

    async def get_user_profile(
        self, oauth_token: CopilotOAuthToken
    ) -> CopilotProfileInfo:
        """Get user profile information from GitHub API.

        Args:
            oauth_token: GitHub OAuth token

        Returns:
            User profile information
        """
        client = await self._get_http_client()

        try:
            # Get basic user info
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {oauth_token.access_token.get_secret_value()}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            response.raise_for_status()
            user_data = response.json()

            # Check Copilot access
            copilot_access = False
            copilot_plan = None

            try:
                copilot_response = await client.get(
                    "https://api.github.com/user/copilot_business_accounts",
                    headers={
                        "Authorization": f"Bearer {oauth_token.access_token.get_secret_value()}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                )
                if copilot_response.status_code == 200:
                    copilot_data = copilot_response.json()
                    copilot_access = (
                        len(copilot_data.get("copilot_business_accounts", [])) > 0
                    )
                    copilot_plan = "business" if copilot_access else None
                elif copilot_response.status_code == 404:
                    # Try individual plan
                    individual_response = await client.get(
                        "https://api.github.com/user/copilot",
                        headers={
                            "Authorization": f"Bearer {oauth_token.access_token.get_secret_value()}",
                            "Accept": "application/vnd.github.v3+json",
                        },
                    )
                    if individual_response.status_code == 200:
                        copilot_access = True
                        copilot_plan = "individual"
            except httpx.HTTPError:
                # Ignore Copilot access check errors
                logger.debug("copilot_access_check_failed")

            profile = CopilotProfileInfo(
                account_id=str(user_data.get("id", user_data["login"])),
                login=user_data["login"],
                name=user_data.get("name"),
                email=user_data.get("email") or "",
                avatar_url=user_data.get("avatar_url"),
                html_url=user_data.get("html_url"),
                copilot_plan=copilot_plan,
                copilot_access=copilot_access,
            )

            logger.info(
                "profile_retrieved",
                login=profile.login,
                user_name=profile.name,
                copilot_access=copilot_access,
                copilot_plan=copilot_plan,
            )

            return profile

        except httpx.HTTPError as e:
            logger.error(
                "profile_request_failed",
                error=str(e),
                status_code=getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                exc_info=e,
            )
            raise

    async def complete_authorization(
        self, device_code: str, interval: int, expires_in: int
    ) -> CopilotCredentials:
        """Complete the full authorization flow.

        Args:
            device_code: Device code from device flow
            interval: Polling interval
            expires_in: Code expiration time

        Returns:
            Complete Copilot credentials
        """
        # Get OAuth token
        oauth_token = await self.poll_for_token(device_code, interval, expires_in)

        # Exchange for Copilot token
        copilot_token = await self.exchange_for_copilot_token(oauth_token)

        # Get user profile
        profile = await self.get_user_profile(oauth_token)

        # Determine account type from profile
        account_type = "individual"
        if profile.copilot_plan == "business":
            account_type = "business"
        elif profile.copilot_plan and "enterprise" in profile.copilot_plan:
            account_type = "enterprise"

        # Create credentials
        credentials = CopilotCredentials(
            oauth_token=oauth_token,
            copilot_token=copilot_token,
            account_type=account_type,
        )

        # Store credentials
        await self.storage.store_credentials(credentials)

        logger.info(
            "authorization_completed",
            login=profile.login,
            account_type=account_type,
            copilot_access=profile.copilot_access,
        )

        return credentials

    async def refresh_copilot_token(
        self, credentials: CopilotCredentials
    ) -> CopilotCredentials:
        """Refresh the Copilot service token using stored OAuth token.

        Args:
            credentials: Current credentials

        Returns:
            Updated credentials with new Copilot token
        """
        if credentials.oauth_token.is_expired:
            logger.warning("oauth_token_expired_cannot_refresh")
            raise ValueError("OAuth token is expired, re-authorization required")

        # Exchange OAuth token for new Copilot token
        new_copilot_token = await self.exchange_for_copilot_token(
            credentials.oauth_token
        )

        # Update credentials
        credentials.copilot_token = new_copilot_token
        credentials.refresh_updated_at()

        # Store updated credentials
        await self.storage.store_credentials(credentials)

        logger.info(
            "copilot_token_refreshed",
            account_type=credentials.account_type,
        )

        return credentials
