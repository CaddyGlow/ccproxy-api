"""Unit tests for the credential balancer token manager."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager
from ccproxy.plugins.credential_balancer.config import (
    CredentialFile,
    CredentialManager,
    CredentialPoolConfig,
    RotationStrategy,
)
from ccproxy.plugins.credential_balancer.manager import (
    CredentialBalancerTokenManager,
    CredentialEntry,
)


async def _write_snapshot(
    path: Path, token: str, *, expires_at: datetime | None = None
) -> None:
    data = {
        "provider": "claude-api",
        "access_token": token,
    }
    if expires_at is not None:
        data["expires_at"] = expires_at.isoformat()
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

    manager = await CredentialBalancerTokenManager.create(pool)

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

    manager = await CredentialBalancerTokenManager.create(pool)

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

    manager = await CredentialBalancerTokenManager.create(pool)

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


@pytest.mark.asyncio
async def test_should_refresh_when_token_expiring(tmp_path: Path) -> None:
    source = tmp_path / "expiring.json"
    await _write_snapshot(
        source,
        "token-expiring",
        expires_at=datetime.now(UTC) + timedelta(seconds=30),
    )

    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="claude_expiring",
        strategy=RotationStrategy.FAILOVER,
        credentials=[CredentialFile(path=source)],
    )

    manager = await CredentialBalancerTokenManager.create(pool)
    credentials = await manager.load_credentials()

    assert manager.should_refresh(credentials, grace_seconds=5.0) is False
    assert manager.should_refresh(credentials, grace_seconds=60.0) is True
    assert manager.should_refresh(credentials) is True


@pytest.mark.asyncio
async def test_should_not_refresh_for_healthy_token(tmp_path: Path) -> None:
    source = tmp_path / "healthy.json"
    await _write_snapshot(
        source,
        "token-healthy",
        expires_at=datetime.now(UTC) + timedelta(hours=2),
    )

    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="claude_healthy",
        strategy=RotationStrategy.FAILOVER,
        credentials=[CredentialFile(path=source)],
    )

    manager = await CredentialBalancerTokenManager.create(pool)
    credentials = await manager.load_credentials()

    assert manager.should_refresh(credentials) is False


@pytest.mark.asyncio
async def test_manager_based_credential_with_mock() -> None:
    """Test balancer with mock AuthManager instances (manager-based credentials)."""
    # Create mock auth managers
    mock_manager_a = AsyncMock(spec=AuthManager)
    mock_manager_a.get_access_token.return_value = "token-manager-a"
    mock_manager_a.is_authenticated.return_value = True
    mock_manager_a.get_provider_name.return_value = "claude-api"

    mock_manager_b = AsyncMock(spec=AuthManager)
    mock_manager_b.get_access_token.return_value = "token-manager-b"
    mock_manager_b.is_authenticated.return_value = True
    mock_manager_b.get_provider_name.return_value = "claude-api"

    # Create credential entries with mock managers
    config_a = CredentialManager(
        manager_key="test-manager-a", label="manager-a", type="manager"
    )
    config_b = CredentialManager(
        manager_key="test-manager-b", label="manager-b", type="manager"
    )

    from ccproxy.core.logging import get_plugin_logger

    logger = get_plugin_logger(__name__)

    entry_a = CredentialEntry(
        config=config_a,
        manager=mock_manager_a,
        max_failures=2,
        cooldown_seconds=60.0,
        logger=logger,
    )

    entry_b = CredentialEntry(
        config=config_b,
        manager=mock_manager_b,
        max_failures=2,
        cooldown_seconds=60.0,
        logger=logger,
    )

    # Create pool config
    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="test_balancer",
        strategy=RotationStrategy.ROUND_ROBIN,
        credentials=[config_a, config_b],
    )

    # Create balancer with pre-created entries
    manager = CredentialBalancerTokenManager(pool, [entry_a, entry_b])

    # Test round-robin behavior
    token_one = await manager.get_access_token()
    assert token_one == "token-manager-a"

    token_two = await manager.get_access_token()
    assert token_two == "token-manager-b"

    token_three = await manager.get_access_token()
    assert token_three == "token-manager-a"


@pytest.mark.asyncio
async def test_manager_based_credential_with_refresh() -> None:
    """Test balancer with mock manager that supports refresh."""
    # Create mock manager with refresh support
    mock_manager = AsyncMock()
    mock_manager.get_access_token.side_effect = [
        AuthenticationError("Token expired"),
        "refreshed-token",
    ]
    mock_manager.get_access_token_with_refresh = AsyncMock(
        return_value="refreshed-token"
    )
    mock_manager.is_authenticated.return_value = True
    mock_manager.get_provider_name.return_value = "claude-api"

    # Create credential entry
    config = CredentialManager(
        manager_key="test-manager", label="with-refresh", type="manager"
    )

    from ccproxy.core.logging import get_plugin_logger

    logger = get_plugin_logger(__name__)

    entry = CredentialEntry(
        config=config,
        manager=mock_manager,
        max_failures=2,
        cooldown_seconds=60.0,
        logger=logger,
    )

    # Create pool config
    pool = CredentialPoolConfig(
        provider="claude-api",
        manager_name="test_refresh_balancer",
        strategy=RotationStrategy.FAILOVER,
        credentials=[config],
    )

    # Create balancer
    manager = CredentialBalancerTokenManager(pool, [entry])

    # Test refresh behavior - first call to get_access_token will raise error
    # But the balancer should handle it by trying refresh
    token = await manager.get_access_token_with_refresh()
    assert token == "refreshed-token"
    mock_manager.get_access_token_with_refresh.assert_called_once()
