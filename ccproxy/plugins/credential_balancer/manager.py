"""Credential rotation manager for the credential balancer plugin."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.core.logging import TraceBoundLogger, get_plugin_logger
from ccproxy.core.request_context import RequestContext

from .config import CredentialPoolConfig, CredentialSource, RotationStrategy


if TYPE_CHECKING:
    from .factory import AuthManagerFactory


logger = get_plugin_logger(__name__)

SNAPSHOT_REFRESH_GRACE_SECONDS = 120.0


@dataclass(slots=True)
class CredentialEntry:
    """Wrapper for an AuthManager with failure tracking and cooldown logic."""

    config: CredentialSource
    manager: AuthManager
    max_failures: int
    cooldown_seconds: float
    logger: TraceBoundLogger
    _failure_count: int = 0
    _disabled_until: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def label(self) -> str:
        """Return a stable label for this credential entry."""
        return self.config.resolved_label

    async def get_access_token(self) -> str:
        """Get access token from the composed manager.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If no valid token available
        """
        async with self._lock:
            return await self.manager.get_access_token()

    async def get_access_token_with_refresh(self) -> str:
        """Get access token with automatic refresh if supported.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If no valid token available
        """
        async with self._lock:
            # Try to use enhanced refresh if available
            if hasattr(self.manager, "get_access_token_with_refresh"):
                return await self.manager.get_access_token_with_refresh()  # type: ignore
            # Fallback to basic get_access_token
            return await self.manager.get_access_token()

    async def is_authenticated(self) -> bool:
        """Check if manager has valid authentication.

        Returns:
            True if authenticated, False otherwise
        """
        try:
            async with self._lock:
                return await self.manager.is_authenticated()
        except Exception:
            return False

    def mark_failure(self) -> None:
        """Record a failure and potentially disable this credential."""
        self._failure_count += 1
        self.logger.debug(
            "credential_balancer_failure_recorded",
            credential=self.label,
            failures=self._failure_count,
        )
        if self._failure_count >= self.max_failures:
            if self.cooldown_seconds > 0:
                self._disabled_until = time.monotonic() + self.cooldown_seconds
            else:
                self._disabled_until = float("inf")
            self.logger.warning(
                "credential_balancer_credential_disabled",
                credential=self.label,
                cooldown_seconds=self.cooldown_seconds,
                failures=self._failure_count,
            )

    def reset_failures(self) -> None:
        """Reset failure count and re-enable this credential."""
        if self._failure_count or self._disabled_until:
            self.logger.debug(
                "credential_balancer_failure_reset",
                credential=self.label,
            )
        self._failure_count = 0
        self._disabled_until = None

    def is_disabled(self, now: float) -> bool:
        """Check if this credential is currently disabled.

        Args:
            now: Current monotonic time

        Returns:
            True if disabled, False if available
        """
        if self._disabled_until is None:
            return False
        if self._disabled_until == float("inf"):
            return True
        if now >= self._disabled_until:
            self.logger.debug(
                "credential_balancer_cooldown_expired",
                credential=self.label,
            )
            self._disabled_until = None
            self._failure_count = 0
            return False
        return True


@dataclass(slots=True)
class _RequestState:
    entry: CredentialEntry
    renew_attempted: bool = False
    created_at: float = field(default_factory=time.monotonic)


class CredentialBalancerTokenManager(AuthManager):
    """Auth manager that rotates across multiple credential sources."""

    def __init__(
        self,
        config: CredentialPoolConfig,
        entries: list[CredentialEntry],
        *,
        logger: TraceBoundLogger | None = None,
    ) -> None:
        """Initialize credential balancer with pre-created entries.

        Args:
            config: Pool configuration
            entries: List of credential entries with composed managers
            logger: Optional logger for this manager
        """
        self._config = config
        self._logger = (logger or get_plugin_logger(__name__)).bind(
            manager=config.manager_name,
            provider=config.provider,
        )
        self._entries = entries
        self._strategy = config.strategy
        self._failure_codes = set(config.failure_status_codes)
        self._lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._request_states: dict[str, _RequestState] = {}
        self._active_index = 0
        self._next_index = 0

    @classmethod
    async def create(
        cls,
        config: CredentialPoolConfig,
        factory: AuthManagerFactory | None = None,
        *,
        logger: TraceBoundLogger | None = None,
    ) -> CredentialBalancerTokenManager:
        """Async factory to create balancer with composed managers.

        Args:
            config: Pool configuration
            factory: Auth manager factory for creating managers from sources
            logger: Optional logger for this manager

        Returns:
            Initialized CredentialBalancerTokenManager instance
        """
        from ccproxy.plugins.credential_balancer.factory import AuthManagerFactory

        if factory is None:
            factory = AuthManagerFactory(logger=logger)

        bound_logger = (logger or get_plugin_logger(__name__)).bind(
            manager=config.manager_name,
            provider=config.provider,
        )

        # Create entries with composed managers
        entries: list[CredentialEntry] = []
        failed_credentials: list[str] = []

        for credential in config.credentials:
            try:
                manager = await factory.create_from_source(credential, config.provider)
                entry = CredentialEntry(
                    config=credential,
                    manager=manager,
                    max_failures=config.max_failures_before_disable,
                    cooldown_seconds=config.cooldown_seconds,
                    logger=bound_logger.bind(credential=credential.resolved_label),
                )
                entries.append(entry)
            except AuthenticationError as e:
                # Log clean warning for failed credential without stack trace
                label = credential.resolved_label
                bound_logger.warning(
                    "credential_balancer_credential_skipped",
                    credential=label,
                    reason=str(e),
                    category="auth",
                )
                failed_credentials.append(label)
                continue
            except Exception as e:
                # Unexpected errors still get logged with type info
                label = credential.resolved_label
                bound_logger.error(
                    "credential_balancer_credential_failed",
                    credential=label,
                    error=str(e),
                    error_type=type(e).__name__,
                    category="auth",
                )
                failed_credentials.append(label)
                continue

        # Warn if some credentials failed
        if failed_credentials:
            bound_logger.warning(
                "credential_balancer_partial_initialization",
                total=len(config.credentials),
                failed=len(failed_credentials),
                succeeded=len(entries),
                failed_labels=failed_credentials,
            )

        # Ensure we have at least one valid credential
        if not entries:
            raise AuthenticationError(
                f"No valid credentials available for {config.manager_name}. "
                f"All {len(config.credentials)} credential(s) failed to load."
            )

        return cls(config, entries, logger=logger)

    async def get_access_token(self) -> str:
        """Get access token from selected credential entry.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If no valid token available
        """
        entry = await self._select_entry()
        try:
            token = await entry.get_access_token()
            request_id = await self._register_request(entry)
            self._logger.debug(
                "credential_balancer_token_selected",
                credential=entry.label,
                request_id=request_id,
            )
            return token
        except AuthenticationError:
            entry.mark_failure()
            await self._handle_entry_failure(entry)
            raise

    async def get_access_token_with_refresh(self) -> str:
        """Get access token with automatic refresh if supported.

        Returns:
            Access token string

        Raises:
            AuthenticationError: If no valid token available
        """
        try:
            return await self.get_access_token()
        except AuthenticationError as exc:
            # Try to refresh the active entry's token
            entry = await self._select_entry(require_active=True)
            try:
                token = await entry.get_access_token_with_refresh()
                request_id = await self._register_request(entry)
                self._logger.debug(
                    "credential_balancer_manual_refresh_succeeded",
                    credential=entry.label,
                    request_id=request_id,
                )
                return token
            except AuthenticationError:
                self._logger.debug(
                    "credential_balancer_manual_refresh_failed",
                    credential=entry.label,
                )
                raise exc

    async def get_credentials(self) -> BaseCredentials:
        raise AuthenticationError(
            "Credential balancer does not expose provider-specific credential models"
        )

    async def is_authenticated(self) -> bool:
        """Check if any credential is authenticated.

        Returns:
            True if at least one credential is authenticated, False otherwise
        """
        try:
            entry = await self._select_entry()
        except AuthenticationError:
            return False
        return await entry.is_authenticated()

    async def get_user_profile(self) -> StandardProfileFields | None:
        """Get user profile (not available for balancer).

        Returns:
            None, as balancer aggregates multiple credentials
        """
        return None

    async def get_profile_quick(self) -> Any:
        """Get profile information without I/O (for compatibility).

        Returns:
            None, as balancer doesn't maintain profile cache
        """
        return None

    async def validate_credentials(self) -> bool:
        """Validate that credentials are available and valid.

        Returns:
            True if valid credentials available, False otherwise
        """
        return await self.is_authenticated()

    def get_provider_name(self) -> str:
        """Get the provider name for this balancer.

        Returns:
            Provider name string
        """
        return self._config.provider

    async def __aenter__(self) -> CredentialBalancerTokenManager:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        return None

    async def load_credentials(self) -> dict[str, TokenSnapshot | None]:
        """Load token snapshots from all credential entries.

        Returns:
            Dictionary mapping credential labels to their token snapshots
        """
        results: dict[str, TokenSnapshot | None] = {}
        for entry in self._entries:
            # Try to get token snapshot from manager if supported
            if hasattr(entry.manager, "get_token_snapshot"):
                try:
                    # Cast to avoid mypy errors with protocol
                    get_snapshot = cast(Any, entry.manager).get_token_snapshot
                    snapshot = cast(TokenSnapshot | None, await get_snapshot())
                    results[entry.label] = snapshot
                except Exception:
                    results[entry.label] = None
            else:
                results[entry.label] = None
        return results

    async def get_token_snapshot(self) -> TokenSnapshot | None:
        """Get token snapshot from selected credential entry.

        Returns:
            TokenSnapshot if available, None otherwise
        """
        entry = await self._select_entry()
        if hasattr(entry.manager, "get_token_snapshot"):
            try:
                # Cast to avoid mypy errors with protocol
                get_snapshot = cast(Any, entry.manager).get_token_snapshot
                return cast(TokenSnapshot | None, await get_snapshot())
            except Exception:
                return None
        return None

    def should_refresh(
        self, credentials: object, grace_seconds: float | None = None
    ) -> bool:
        snapshots: list[TokenSnapshot] = []
        if isinstance(credentials, dict):
            for value in credentials.values():
                if value is None:
                    return True
                if isinstance(value, TokenSnapshot):
                    snapshots.append(value)
        elif isinstance(credentials, TokenSnapshot):
            snapshots = [credentials]
        else:
            return False

        if not snapshots:
            return False

        threshold = (
            SNAPSHOT_REFRESH_GRACE_SECONDS
            if grace_seconds is None
            else max(grace_seconds, 0.0)
        )

        now = datetime.now(UTC)
        for snapshot in snapshots:
            expires_at = snapshot.expires_at
            if expires_at is None:
                continue
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            remaining = (expires_at - now).total_seconds()
            if remaining <= threshold:
                return True

        return any(not snapshot.access_token for snapshot in snapshots)

    async def handle_response_event(
        self, request_id: str | None, status_code: int | None
    ) -> bool:
        if not request_id:
            return False

        async with self._state_lock:
            state = self._request_states.pop(request_id, None)
        if state is None:
            return False

        entry = state.entry
        if status_code is None:
            self._logger.debug(
                "credential_balancer_event_without_status",
                credential=entry.label,
                request_id=request_id,
            )
            return True

        if status_code < 400:
            entry.reset_failures()
            return True

        if status_code not in self._failure_codes:
            return True

        self._logger.warning(
            "credential_balancer_failure_detected",
            credential=entry.label,
            request_id=request_id,
            status_code=status_code,
        )

        entry.mark_failure()
        await self._handle_entry_failure(entry)
        return True

    async def cleanup_expired_requests(self, max_age_seconds: float = 120.0) -> None:
        cutoff = time.monotonic() - max_age_seconds
        async with self._state_lock:
            stale = [
                key
                for key, value in self._request_states.items()
                if value.created_at < cutoff
            ]
            for key in stale:
                del self._request_states[key]

    async def _register_request(self, entry: CredentialEntry) -> str:
        request_id: str | None = None
        context = RequestContext.get_current()
        if context is not None:
            request_id = getattr(context, "request_id", None)
        if not request_id:
            request_id = f"cred-{uuid.uuid4()}"

        state = _RequestState(entry=entry)
        async with self._state_lock:
            self._request_states[request_id] = state
        return request_id

    async def _select_entry(self, *, require_active: bool = False) -> CredentialEntry:
        """Select an available credential entry based on strategy.

        Args:
            require_active: If True, start with the active entry (for failover)

        Returns:
            Selected CredentialEntry

        Raises:
            AuthenticationError: If no credentials available
        """
        if not self._entries:
            raise AuthenticationError("No credentials configured")

        async with self._lock:
            total = len(self._entries)
            if require_active and self._strategy == RotationStrategy.FAILOVER:
                indices = [self._active_index] + [
                    (self._active_index + offset) % total for offset in range(1, total)
                ]
            elif self._strategy == RotationStrategy.ROUND_ROBIN:
                start = self._next_index
                self._next_index = (self._next_index + 1) % total
                indices = [(start + offset) % total for offset in range(total)]
            else:
                start = self._active_index
                indices = [(start + offset) % total for offset in range(total)]

        now = time.monotonic()
        last_error: Exception | None = None
        for idx in indices:
            entry = self._entries[idx]
            if entry.is_disabled(now):
                continue

            # Check if entry is authenticated using composed manager
            is_auth = await entry.is_authenticated()
            if not is_auth:
                entry.mark_failure()
                last_error = AuthenticationError("Credential not authenticated")
                continue

            if self._strategy == RotationStrategy.FAILOVER:
                async with self._lock:
                    self._active_index = idx
            return entry

        if last_error:
            raise last_error
        raise AuthenticationError("No credential is currently available")

    async def _handle_entry_failure(self, entry: CredentialEntry) -> None:
        if self._strategy != RotationStrategy.FAILOVER:
            return
        async with self._lock:
            current = self._active_index
            if self._entries[current] is entry:
                self._active_index = (current + 1) % len(self._entries)
                self._logger.info(
                    "credential_balancer_failover",
                    previous=entry.label,
                    next=self._entries[self._active_index].label,
                )


__all__ = ["CredentialBalancerTokenManager", "CredentialEntry"]
