"""Copilot token manager implementation."""

from __future__ import annotations

from datetime import UTC, datetime
from time import time
from typing import Any

import httpx

from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger

from .config import CopilotOAuthConfig
from .oauth.client import CopilotOAuthClient
from .oauth.models import CopilotCredentials
from .oauth.storage import CopilotOAuthStorage


logger = get_plugin_logger()


class CopilotTokenManager(BaseTokenManager[CopilotCredentials]):
    """Manager for GitHub Copilot credential lifecycle."""

    def __init__(
        self,
        storage: TokenStorage[CopilotCredentials] | None = None,
        *,
        config: CopilotOAuthConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: Any | None = None,
    ) -> None:
        storage = storage or CopilotOAuthStorage()
        super().__init__(storage)
        self.config = config or CopilotOAuthConfig()
        self._client = CopilotOAuthClient(
            self.config,
            storage
            if isinstance(storage, CopilotOAuthStorage)
            else CopilotOAuthStorage(),
            http_client=http_client,
            hook_manager=hook_manager,
            detection_service=detection_service,
        )
        self._profile_cache: StandardProfileFields | None = None

    @classmethod
    async def create(
        cls,
        storage: TokenStorage[CopilotCredentials] | None = None,
        *,
        config: CopilotOAuthConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
        hook_manager: Any | None = None,
        detection_service: Any | None = None,
    ) -> CopilotTokenManager:
        """Async factory for parity with other managers."""
        return cls(
            storage=storage,
            config=config,
            http_client=http_client,
            hook_manager=hook_manager,
            detection_service=detection_service,
        )

    def _build_token_snapshot(self, credentials: CopilotCredentials) -> TokenSnapshot:
        """Construct a token snapshot for Copilot credentials."""
        access_token: str | None = None
        copilot_token = credentials.copilot_token
        if copilot_token and copilot_token.token:
            access_token = copilot_token.token.get_secret_value()

        refresh_token: str | None = None
        oauth_token = credentials.oauth_token
        if oauth_token.refresh_token:
            refresh_token = oauth_token.refresh_token.get_secret_value()

        expires_at = None
        if copilot_token and copilot_token.expires_at:
            expires_at = copilot_token.expires_at
        else:
            if oauth_token.expires_in and oauth_token.created_at:
                expires_at = oauth_token.expires_at_datetime

        scope_value = oauth_token.scope or ""
        scopes = tuple(
            scope
            for scope in (item.strip() for item in scope_value.split(" "))
            if scope
        )

        extras = {
            "account_type": credentials.account_type,
            "has_copilot_token": bool(credentials.copilot_token),
        }

        logger.debug(
            "copilot_token_snapshot",
            scopes=scopes,
            expires_at=expires_at,
            credentials=credentials,
            access_token=access_token,
            refresh_token=refresh_token,
        )
        return TokenSnapshot(
            provider="copilot",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            scopes=scopes,
            extras=extras,
        )

    # ==================================================================
    # BaseTokenManager protocol implementations
    # ==================================================================

    async def refresh_token(self) -> CopilotCredentials | None:
        credentials = await self.load_credentials()
        if not credentials:
            logger.error("copilot_refresh_no_credentials", category="auth")
            return None

        try:
            refreshed = await self._client.refresh_copilot_token(credentials)
            # Client already persisted credentials; refresh in-memory caches.
            self._credentials_cache = refreshed
            self._credentials_loaded_at = time()
            self._auth_cache.clear()
            self._profile_cache = None
            return refreshed
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "copilot_refresh_failed",
                error=str(exc),
                exc_info=exc,
                category="auth",
            )
            return None

    def is_expired(self, credentials: CopilotCredentials) -> bool:
        token = credentials.copilot_token
        if not token:
            return True

        now = datetime.now(UTC)
        if token.expires_at and now >= token.expires_at:
            return True

        refresh_deadline = self._compute_refresh_deadline(credentials)
        if refresh_deadline and now >= refresh_deadline:
            return True

        return credentials.oauth_token.is_expired

    def get_account_id(self, credentials: CopilotCredentials) -> str | None:
        # GitHub account information is part of profile, not raw credentials.
        return None

    def get_expiration_time(self, credentials: CopilotCredentials) -> datetime | None:
        candidates: list[datetime] = []

        token = credentials.copilot_token
        if token:
            if token.expires_at:
                candidates.append(token.expires_at)

            refresh_deadline = self._compute_refresh_deadline(credentials)
            if refresh_deadline:
                candidates.append(refresh_deadline)

        if credentials.oauth_token.expires_in and credentials.oauth_token.created_at:
            candidates.append(credentials.oauth_token.expires_at_datetime)

        if not candidates:
            return None

        return min(candidates)

    # ==================================================================
    # Token access helpers used by adapters/routes
    # ==================================================================

    async def ensure_copilot_token(self) -> str:
        credentials = await self.load_credentials()
        if not credentials:
            raise ValueError("No Copilot credentials available")

        if credentials.oauth_token.is_expired:
            raise ValueError("OAuth token expired; re-authentication required")

        if not credentials.copilot_token or credentials.copilot_token.is_expired:
            logger.debug("copilot_token_refresh_needed", category="auth")
            credentials = await self._client.refresh_copilot_token(credentials)
            self._credentials_cache = credentials
            self._credentials_loaded_at = time()
            self._auth_cache.clear()
            self._profile_cache = None

        token = credentials.copilot_token
        if not token:
            raise ValueError("Unable to obtain Copilot service token")
        return token.token.get_secret_value()

    async def ensure_oauth_token(self) -> str:
        credentials = await self.load_credentials()
        if not credentials:
            raise ValueError("No Copilot credentials available")
        if credentials.oauth_token.is_expired:
            raise ValueError("OAuth token expired; re-authentication required")
        return credentials.oauth_token.access_token.get_secret_value()

    async def get_access_token(self) -> str | None:
        try:
            return await self.ensure_copilot_token()
        except Exception as exc:
            logger.error(
                "copilot_access_token_failed",
                error=str(exc),
                category="auth",
            )
            return None

    async def get_access_token_with_refresh(self) -> str | None:
        return await self.get_access_token()

    async def get_profile(self) -> StandardProfileFields | None:
        if self._profile_cache:
            return self._profile_cache
        credentials = await self.load_credentials()
        if not credentials:
            return None
        try:
            profile = await self._client.get_standard_profile(credentials.oauth_token)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("copilot_profile_fetch_failed", error=str(exc))
            return None
        self._profile_cache = profile
        return profile

    async def get_profile_quick(self) -> StandardProfileFields | None:
        return await self.get_profile()

    async def aclose(self) -> None:
        await self._client.close()

    def _compute_refresh_deadline(
        self, credentials: CopilotCredentials
    ) -> datetime | None:
        token = credentials.copilot_token
        if not token or not token.refresh_in:
            return None

        try:
            updated_at = int(credentials.updated_at)
        except (TypeError, ValueError):
            return None

        try:
            refresh_in = int(token.refresh_in)
        except (TypeError, ValueError):
            return None

        if refresh_in <= 0:
            return datetime.now(UTC)

        return datetime.fromtimestamp(updated_at + refresh_in, tz=UTC)


__all__ = ["CopilotTokenManager"]
