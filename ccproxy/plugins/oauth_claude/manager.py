"""Claude API token manager implementation for the Claude API plugin."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast

import httpx


if TYPE_CHECKING:
    pass

from ccproxy.auth.managers.base_enhanced import EnhancedTokenManager
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger

from .config import ClaudeOAuthConfig
from .models import ClaudeCredentials, ClaudeProfileInfo, ClaudeTokenWrapper
from .storage import ClaudeOAuthStorage, ClaudeProfileStorage


class TokenRefreshProvider(Protocol):
    """Protocol for token refresh capability."""

    async def refresh_access_token(self, refresh_token: str) -> ClaudeCredentials:
        """Refresh access token using refresh token."""
        ...


logger = get_plugin_logger()


class ClaudeApiTokenManager(EnhancedTokenManager[ClaudeCredentials]):
    """Manager for Claude API token storage and refresh operations.

    Uses the Claude-specific storage implementation with enhanced token management.
    """

    def __init__(
        self,
        storage: TokenStorage[ClaudeCredentials] | None = None,
        http_client: "httpx.AsyncClient | None" = None,
        oauth_provider: TokenRefreshProvider | None = None,
    ):
        """Initialize Claude API token manager.

        Args:
            storage: Optional custom storage, defaults to standard location
            http_client: Optional HTTP client for API requests
            oauth_provider: Optional OAuth provider for token refresh (protocol injection)
        """
        if storage is None:
            storage = ClaudeOAuthStorage()
        super().__init__(storage)
        self._profile_cache: ClaudeProfileInfo | None = None
        self.oauth_provider = oauth_provider

        # Create default HTTP client if not provided; track ownership
        self._owns_client = False
        if http_client is None:
            http_client = httpx.AsyncClient()
            self._owns_client = True
        self.http_client = http_client

    # ==================== Internal helpers ====================

    def _derive_subscription_type(self, profile: "ClaudeProfileInfo") -> str:
        """Derive subscription type string from profile info.

        Priority: "max" > "pro" > "free".
        """
        try:
            if getattr(profile, "has_claude_max", None):
                return "max"
            if getattr(profile, "has_claude_pro", None):
                return "pro"
            return "free"
        except Exception:
            # Be defensive; default to free if unexpected structure
            return "free"

    async def _sync_subscription_type_with_profile(
        self,
        profile: "ClaudeProfileInfo",
        credentials: "ClaudeCredentials | None" = None,
    ) -> None:
        """Update stored credentials with subscription type from profile.

        Avoids unnecessary writes by only saving when the value changes.
        If credentials are not provided, they will be loaded once.
        """
        try:
            new_sub = self._derive_subscription_type(profile)

            # Use provided credentials to avoid an extra read if available
            creds = credentials or await self.load_credentials()
            if not creds or not hasattr(creds, "claude_ai_oauth"):
                return

            current_sub = creds.claude_ai_oauth.subscription_type
            if current_sub != new_sub:
                creds.claude_ai_oauth.subscription_type = new_sub
                await self.save_credentials(creds)
                logger.info(
                    "claude_subscription_type_updated",
                    subscription_type=new_sub,
                    category="auth",
                )
        except Exception as e:
            # Non-fatal: syncing subscription type should never break profile flow
            logger.debug(
                "claude_subscription_type_update_failed",
                error=str(e),
                category="auth",
            )

    @classmethod
    async def create(
        cls,
        storage: TokenStorage["ClaudeCredentials"] | None = None,
        http_client: "httpx.AsyncClient | None" = None,
        oauth_provider: TokenRefreshProvider | None = None,
    ) -> "ClaudeApiTokenManager":
        """Async factory that constructs the manager and preloads cached profile.

        This avoids creating event loops in __init__ and keeps initialization non-blocking.
        """
        manager = cls(
            storage=storage, http_client=http_client, oauth_provider=oauth_provider
        )
        await manager.preload_profile_cache()
        return manager

    def _build_token_snapshot(self, credentials: ClaudeCredentials) -> TokenSnapshot:
        """Construct a token snapshot for Claude credentials."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        scopes = tuple(wrapper.scopes)
        extras = {
            "subscription_type": wrapper.subscription_type,
        }
        return TokenSnapshot(
            provider="claude-api",
            access_token=str(wrapper.access_token_value),
            refresh_token=wrapper.refresh_token_value,
            expires_at=wrapper.expires_at_datetime,
            scopes=scopes,
            extras=extras,
        )

    async def preload_profile_cache(self) -> None:
        """Load profile from storage asynchronously if available."""
        try:
            profile_storage = ClaudeProfileStorage()

            # Only attempt to read if the file exists
            if profile_storage.file_path.exists():
                profile = await profile_storage.load_profile()
                if profile:
                    self._profile_cache = profile
                    logger.debug(
                        "claude_profile_loaded_from_cache",
                        account_id=profile.account_id,
                        email=profile.email,
                        category="auth",
                    )
        except Exception as e:
            # Don't fail if profile can't be loaded
            logger.debug(
                "claude_profile_cache_load_failed",
                error=str(e),
                category="auth",
            )

    # ==================== Enhanced Token Management Methods ====================

    async def get_access_token(self) -> str:
        """Get access token using enhanced base with automatic refresh."""
        token = await self.get_access_token_with_refresh()
        if not token:
            from ccproxy.auth.exceptions import CredentialsInvalidError

            raise CredentialsInvalidError("No valid access token available")
        return token

    async def refresh_token_if_needed(self) -> ClaudeCredentials | None:
        """Use enhanced base's automatic refresh capability."""
        if await self.ensure_valid_token():
            return await self.load_credentials()
        return None

    # ==================== Abstract Method Implementations ====================

    async def refresh_token(self) -> ClaudeCredentials | None:
        """Refresh the access token using the refresh token.

        Returns:
            Updated credentials or None if refresh failed
        """
        # Load current credentials and extract refresh token
        credentials = await self.load_credentials()
        if not credentials:
            logger.error("no_credentials_to_refresh", category="auth")
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        refresh_token = wrapper.refresh_token_value
        if not refresh_token:
            logger.error("no_refresh_token_available", category="auth")
            return None

        try:
            # Use injected provider or fallback to local import
            new_credentials: ClaudeCredentials
            if self.oauth_provider:
                new_credentials = await self.oauth_provider.refresh_access_token(
                    refresh_token
                )
            else:
                # Fallback to local import if no provider injected
                from .provider import ClaudeOAuthProvider

                provider = ClaudeOAuthProvider(http_client=self.http_client)
                new_credentials = await provider.refresh_access_token(refresh_token)

            # Save updated credentials
            if await self.save_credentials(new_credentials):
                logger.info("token_refreshed_successfully", category="auth")
                # Clear profile cache as token changed
                self._profile_cache = None

                return new_credentials

            logger.error("failed_to_save_refreshed_credentials", category="auth")
            return None

        except Exception as e:
            logger.error(
                "Token refresh failed",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    def is_expired(self, credentials: ClaudeCredentials) -> bool:
        """Check if credentials are expired using wrapper."""
        if isinstance(credentials, ClaudeCredentials):
            wrapper = ClaudeTokenWrapper(credentials=credentials)
            return bool(wrapper.is_expired)

        expires_at = getattr(credentials, "expires_at", None)
        if expires_at is None:
            expires_at = getattr(credentials, "claude_ai_oauth", None)
            if expires_at is not None:
                expires_at = getattr(expires_at, "expires_at", None)

        if expires_at is None:
            return False

        if isinstance(expires_at, datetime):
            return expires_at <= datetime.now(UTC)
        if isinstance(expires_at, int | float):
            return datetime.fromtimestamp(expires_at / 1000, tz=UTC) <= datetime.now(
                UTC
            )

        return False

    # ==================== Targeted overrides ====================

    async def load_credentials(self) -> ClaudeCredentials | None:
        """Load credentials and backfill subscription_type from profile if missing.

        Avoids network calls; uses cached profile or local ~/.claude/.account.json
        and writes back only when the field actually changes.
        """
        creds = await super().load_credentials()
        if not creds or not hasattr(creds, "claude_ai_oauth"):
            return creds

        sub = creds.claude_ai_oauth.subscription_type
        if sub is None or str(sub).strip().lower() in {"", "unknown"}:
            # Try cached profile first to avoid an extra file read
            profile: ClaudeProfileInfo | None = self._profile_cache
            if profile is None:
                # Only read from disk if the profile file exists; no API calls here
                try:
                    profile_storage = ClaudeProfileStorage()
                    if profile_storage.file_path.exists():
                        profile = await profile_storage.load_profile()
                        if profile:
                            self._profile_cache = profile
                except Exception:
                    profile = None

            if profile is not None:
                try:
                    new_sub = self._derive_subscription_type(profile)
                    if new_sub != sub:
                        creds.claude_ai_oauth.subscription_type = new_sub
                        await self.save_credentials(creds)
                        logger.info(
                            "claude_subscription_type_backfilled_on_load",
                            subscription_type=new_sub,
                            category="auth",
                        )
                except Exception as e:
                    logger.debug(
                        "claude_subscription_type_backfill_failed",
                        error=str(e),
                        category="auth",
                    )

        return creds

    def get_account_id(self, credentials: ClaudeCredentials) -> str | None:
        """Get account ID from credentials.

        Claude doesn't store account_id in tokens, would need
        to fetch from profile API.
        """
        if self._profile_cache:
            return self._profile_cache.account_id
        return None

    # ==================== Claude-Specific Methods ====================

    def get_expiration_time(self, credentials: ClaudeCredentials) -> datetime | None:
        """Get expiration time as datetime."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.expires_at_datetime

    async def get_profile_quick(self) -> ClaudeProfileInfo | None:
        """Return cached profile info only, avoiding I/O or network.

        Profile cache is typically preloaded from local storage by
        the async factory create() via preload_profile_cache().

        Returns:
            Cached ClaudeProfileInfo or None
        """
        return self._profile_cache

    async def get_access_token_value(self) -> str | None:
        """Get the actual access token value.

        Returns:
            Access token string if available, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if self.is_expired(credentials):
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return cast(str, wrapper.access_token_value)

    async def get_profile(self) -> ClaudeProfileInfo | None:
        """Get user profile from cache or API.

        Returns:
            ClaudeProfileInfo or None if not authenticated
        """
        if self._profile_cache:
            return self._profile_cache

        # Try to load from .account.json first

        profile_storage = ClaudeProfileStorage()
        profile = await profile_storage.load_profile()
        if profile:
            self._profile_cache = profile
            # Best-effort sync of subscription type from cached profile
            await self._sync_subscription_type_with_profile(profile)
            return profile

        # If not in storage, fetch from API
        credentials = await self.load_credentials()
        if not credentials or self.is_expired(credentials):
            return None

        # Get access token
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        access_token = cast(str, wrapper.access_token_value)
        if not access_token:
            return None

        # Fetch profile from API and save
        try:
            config = ClaudeOAuthConfig()

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            # Optionally add detection headers if client supports it
            try:
                # Use injected provider or fallback to local import
                if self.oauth_provider and hasattr(self.oauth_provider, "client"):
                    if hasattr(self.oauth_provider.client, "get_custom_headers"):
                        headers.update(self.oauth_provider.client.get_custom_headers())
                else:
                    # Fallback to local import if no provider injected
                    from .provider import ClaudeOAuthProvider

                    temp_provider = ClaudeOAuthProvider(http_client=self.http_client)
                    if hasattr(temp_provider, "client") and hasattr(
                        temp_provider.client, "get_custom_headers"
                    ):
                        headers.update(temp_provider.client.get_custom_headers())
            except Exception:
                pass

            # Debug logging for HTTP client usage
            logger.debug(
                "claude_manager_making_http_request",
                url=config.profile_url,
                http_client_id=id(self.http_client),
                has_hooks=hasattr(self.http_client, "hook_manager")
                and self.http_client.hook_manager is not None,
                hook_manager_id=id(self.http_client.hook_manager)
                if hasattr(self.http_client, "hook_manager")
                and self.http_client.hook_manager
                else None,
            )

            # Use the injected HTTP client
            response = await self.http_client.get(
                config.profile_url,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            profile_data = response.json()

            # Save to .account.json
            await profile_storage.save_profile(profile_data)

            # Parse and cache
            profile = ClaudeProfileInfo.from_api_response(profile_data)
            self._profile_cache = profile

            # Sync subscription type to credentials in a single write if changed
            await self._sync_subscription_type_with_profile(
                profile, credentials=credentials
            )

            logger.info(
                "claude_profile_fetched_from_api",
                account_id=profile.account_id,
                email=profile.email,
                category="auth",
            )

            return profile

        except Exception as e:
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(
                    "claude_profile_api_error",
                    status_code=e.response.status_code,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
            else:
                logger.error(
                    "claude_profile_fetch_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                    category="auth",
                )
            return None

    async def close(self) -> None:
        """Close the HTTP client if it was created internally."""
        if getattr(self, "_owns_client", False) and self.http_client:
            await self.http_client.aclose()
