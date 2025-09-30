"""Credential rotation manager for the credential balancer plugin."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager
from ccproxy.auth.managers.token_snapshot import TokenSnapshot
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.core.logging import TraceBoundLogger, get_plugin_logger
from ccproxy.core.request_context import RequestContext

from .config import CredentialFile, CredentialPoolConfig, RotationStrategy


logger = get_plugin_logger(__name__)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


async def _read_snapshot(path: Path) -> dict[str, Any] | None:
    def _load() -> dict[str, Any] | None:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            logger.warning("credential_balancer_snapshot_not_mapping", path=str(path))
            return None
        return cast(dict[str, Any], data)

    try:
        return await asyncio.to_thread(_load)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.warning(
            "credential_balancer_invalid_json", path=str(path), error=str(exc)
        )
        return None
    except PermissionError as exc:
        logger.warning(
            "credential_balancer_permission_denied", path=str(path), error=str(exc)
        )
        return None
    except OSError as exc:
        logger.warning(
            "credential_balancer_read_failed", path=str(path), error=str(exc)
        )
        return None


def _fingerprint_snapshot(snapshot: TokenSnapshot | None) -> str | None:
    if snapshot is None:
        return None
    token = snapshot.access_token or ""
    refresh = snapshot.refresh_token or ""
    expires = snapshot.expires_at.isoformat() if snapshot.expires_at else ""
    return "|".join((snapshot.provider or "", token, refresh, expires))


@dataclass(slots=True)
class CredentialEntry:
    config: CredentialFile
    max_failures: int
    cooldown_seconds: float
    provider: str
    logger: TraceBoundLogger
    _snapshot: TokenSnapshot | None = None
    _fingerprint: str | None = None
    _failure_count: int = 0
    _disabled_until: float | None = None
    _mtime_ns: int | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def label(self) -> str:
        """Return a stable label for this credential entry."""

        return self.config.resolved_label

    async def ensure_snapshot(self, *, force: bool = False) -> TokenSnapshot | None:
        async with self._lock:
            path = self.config.path

            try:
                stat_result = await asyncio.to_thread(path.stat)
                mtime_ns = stat_result.st_mtime_ns
            except FileNotFoundError:
                self._snapshot = None
                self._fingerprint = None
                self._mtime_ns = None
                return None
            except PermissionError as exc:
                self.logger.warning(
                    "credential_balancer_stat_permission_denied",
                    path=str(path),
                    error=str(exc),
                )
                return self._snapshot
            except OSError as exc:
                self.logger.warning(
                    "credential_balancer_stat_failed",
                    path=str(path),
                    error=str(exc),
                )
                return self._snapshot

            need_reload = force or self._snapshot is None or mtime_ns != self._mtime_ns
            if not need_reload:
                return self._snapshot

            raw = await _read_snapshot(path)
            if raw is None:
                self._snapshot = None
                self._fingerprint = None
                self._mtime_ns = mtime_ns
                return None

            snapshot = TokenSnapshot(
                provider=str(raw.get("provider", self.provider)),
                account_id=raw.get("account_id"),
                access_token=raw.get("access_token"),
                refresh_token=raw.get("refresh_token"),
                expires_at=_parse_datetime(raw.get("expires_at")),
                scopes=tuple(raw.get("scopes", []) if raw.get("scopes") else ()),
                extras=dict(raw.get("extras", {}))
                if isinstance(raw.get("extras"), dict)
                else {},
            )

            self._snapshot = snapshot
            new_fingerprint = _fingerprint_snapshot(snapshot)
            if new_fingerprint != self._fingerprint:
                self.logger.info(
                    "credential_balancer_snapshot_loaded",
                    credential=self.config.label,
                    provider=snapshot.provider,
                )
                self._fingerprint = new_fingerprint
                self._failure_count = 0
                self._disabled_until = None
            self._mtime_ns = mtime_ns
            return snapshot

    def mark_failure(self) -> None:
        self._failure_count += 1
        self.logger.debug(
            "credential_balancer_failure_recorded",
            credential=self.config.label,
            failures=self._failure_count,
        )
        if self._failure_count >= self.max_failures:
            if self.cooldown_seconds > 0:
                self._disabled_until = time.monotonic() + self.cooldown_seconds
            else:
                self._disabled_until = float("inf")
            self.logger.warning(
                "credential_balancer_credential_disabled",
                credential=self.config.label,
                cooldown_seconds=self.cooldown_seconds,
                failures=self._failure_count,
            )

    def reset_failures(self) -> None:
        if self._failure_count or self._disabled_until:
            self.logger.debug(
                "credential_balancer_failure_reset",
                credential=self.config.label,
            )
        self._failure_count = 0
        self._disabled_until = None

    def is_disabled(self, now: float) -> bool:
        if self._disabled_until is None:
            return False
        if self._disabled_until == float("inf"):
            return True
        if now >= self._disabled_until:
            self.logger.debug(
                "credential_balancer_cooldown_expired",
                credential=self.config.label,
            )
            self._disabled_until = None
            self._failure_count = 0
            return False
        return True

    def has_token(self) -> bool:
        return bool(self._snapshot and self._snapshot.access_token)

    async def refresh_snapshot(self) -> bool:
        before = self._fingerprint
        snapshot = await self.ensure_snapshot(force=True)
        after = _fingerprint_snapshot(snapshot)
        changed = before != after
        if changed:
            self.logger.info(
                "credential_balancer_snapshot_refreshed",
                credential=self.config.label,
            )
        return changed and self.has_token()


@dataclass(slots=True)
class _RequestState:
    entry: CredentialEntry
    renew_attempted: bool = False
    created_at: float = field(default_factory=time.monotonic)


class CredentialBalancerTokenManager(AuthManager):
    """Auth manager that rotates across multiple credential files."""

    def __init__(
        self,
        config: CredentialPoolConfig,
        *,
        logger: TraceBoundLogger | None = None,
    ) -> None:
        self._config = config
        self._logger = (logger or get_plugin_logger(__name__)).bind(
            manager=config.manager_name,
            provider=config.provider,
        )
        self._entries: list[CredentialEntry] = [
            CredentialEntry(
                credential,
                max_failures=config.max_failures_before_disable,
                cooldown_seconds=config.cooldown_seconds,
                provider=config.provider,
                logger=self._logger.bind(credential=credential.resolved_label),
            )
            for credential in config.credentials
        ]
        self._strategy = config.strategy
        self._failure_codes = set(config.failure_status_codes)
        self._lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._request_states: dict[str, _RequestState] = {}
        self._active_index = 0
        self._next_index = 0

    async def get_access_token(self) -> str:
        entry = await self._select_entry()
        snapshot = await entry.ensure_snapshot()
        if not snapshot or not snapshot.access_token:
            entry.mark_failure()
            await self._handle_entry_failure(entry)
            raise AuthenticationError("No valid access token available")

        request_id = await self._register_request(entry)
        self._logger.debug(
            "credential_balancer_token_selected",
            credential=entry.label,
            request_id=request_id,
        )
        return str(snapshot.access_token)

    async def get_access_token_with_refresh(self) -> str:
        try:
            return await self.get_access_token()
        except AuthenticationError as exc:
            entry = await self._select_entry(require_active=True)
            changed = await entry.refresh_snapshot()
            if changed and entry.has_token():
                snapshot = entry._snapshot
                request_id = await self._register_request(entry)
                self._logger.debug(
                    "credential_balancer_manual_refresh_succeeded",
                    credential=entry.label,
                    request_id=request_id,
                )
                if snapshot and snapshot.access_token:
                    return str(snapshot.access_token)
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
        try:
            entry = await self._select_entry()
        except AuthenticationError:
            return False
        snapshot = await entry.ensure_snapshot()
        return bool(snapshot and snapshot.access_token)

    async def get_user_profile(self) -> StandardProfileFields | None:
        return None

    async def validate_credentials(self) -> bool:
        return await self.is_authenticated()

    def get_provider_name(self) -> str:
        return self._config.provider

    async def __aenter__(self) -> CredentialBalancerTokenManager:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def load_credentials(self) -> dict[str, TokenSnapshot | None]:
        results: dict[str, TokenSnapshot | None] = {}
        for entry in self._entries:
            results[entry.label] = await entry.ensure_snapshot()
        return results

    async def get_token_snapshot(self) -> TokenSnapshot | None:
        entry = await self._select_entry()
        return await entry.ensure_snapshot()

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
            snapshot = await entry.ensure_snapshot()
            if not snapshot or not snapshot.access_token:
                entry.mark_failure()
                last_error = AuthenticationError("Credential missing access token")
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
