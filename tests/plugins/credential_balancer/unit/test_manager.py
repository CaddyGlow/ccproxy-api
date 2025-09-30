"""Unit tests for the credential balancer token manager."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.plugins.credential_balancer.config import (
    CredentialFile,
    CredentialPoolConfig,
    RotationStrategy,
)
from ccproxy.plugins.credential_balancer.manager import CredentialBalancerTokenManager


async def _write_snapshot(path: Path, token: str) -> None:
    data = {
        "provider": "claude-api",
        "access_token": token,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    await asyncio.sleep(0)


def _pop_request_id(manager: CredentialBalancerTokenManager) -> str:
    assert manager._request_states, "expected pending request state"
    return next(iter(manager._request_states.keys()))


@pytest.mark.asyncio
async def test_round_robin_rotation(tmp_path: Path) -> None:
    file_a = tmp_path / "cred_a.json"
    file_b = tmp_path / "cred_b.json"
    await _write_snapshot(file_a, "token-a")
    await _write_snapshot(file_b, "token-b")

    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="claude_balancer",
        strategy=RotationStrategy.ROUND_ROBIN,
        credentials=[CredentialFile(path=file_a), CredentialFile(path=file_b)],
    )

    manager = CredentialBalancerTokenManager(pool)

    token_one = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 200)

    token_two = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 200)

    token_three = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 200)

    assert token_one == "token-a"
    assert token_two == "token-b"
    assert token_three == "token-a"


@pytest.mark.asyncio
async def test_failover_after_failure(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    backup = tmp_path / "backup.json"
    await _write_snapshot(primary, "token-primary")
    await _write_snapshot(backup, "token-backup")

    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="claude_failover",
        strategy=RotationStrategy.FAILOVER,
        credentials=[CredentialFile(path=primary), CredentialFile(path=backup)],
        max_failures_before_disable=1,
    )

    manager = CredentialBalancerTokenManager(pool)

    token_first = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 401)

    token_second = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 200)

    assert token_first == "token-primary"
    assert token_second == "token-backup"


@pytest.mark.asyncio
async def test_failure_does_not_reload_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "renew.json"
    await _write_snapshot(source, "token-old")

    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="claude_renew",
        strategy=RotationStrategy.FAILOVER,
        credentials=[CredentialFile(path=source)],
        max_failures_before_disable=1,
        cooldown_seconds=0.0,
    )

    manager = CredentialBalancerTokenManager(pool)

    token_one = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)

    await asyncio.sleep(0.05)
    await _write_snapshot(source, "token-new")

    await manager.handle_response_event(request_id, 401)

    with pytest.raises(AuthenticationError):
        await asyncio.wait_for(manager.get_access_token(), timeout=5)

    # Even though the file on disk now contains a new token, the balancer does
    # not reload automatically; the credential remains disabled until cooldown.
    with pytest.raises(AuthenticationError):
        await manager.get_token_snapshot()
