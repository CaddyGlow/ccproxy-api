"""Unit tests for the credential balancer token manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.plugins.credential_balancer.config import (
    CredentialManager,
    CredentialPoolConfig,
    RotationStrategy,
)
from ccproxy.plugins.credential_balancer.factory import AuthManagerFactory
from ccproxy.plugins.credential_balancer.manager import (
    CredentialBalancerTokenManager,
)


# Simple test auth manager for testing (rename to avoid pytest collection warning)
class MockAuthManager:
    """Test auth manager that returns a static token."""

    def __init__(self, token: str):
        self._token = token

    async def get_access_token(self) -> str:
        return self._token

    async def get_credentials(self) -> None:
        return None

    async def is_authenticated(self) -> bool:
        return True

    async def get_user_profile(self) -> StandardProfileFields | None:
        return None

    async def validate_credentials(self) -> bool:
        return True

    def get_provider_name(self) -> str:
        return "test"

    async def __aenter__(self) -> MockAuthManager:
        return self

    async def __aexit__(self, *args) -> None:
        pass


def _pop_request_id(manager: CredentialBalancerTokenManager) -> str:
    assert manager._request_states, "expected pending request state"
    return next(iter(manager._request_states.keys()))


@pytest.mark.asyncio
async def test_round_robin_rotation() -> None:
    """Test that round-robin strategy rotates between credentials."""
    pool = CredentialPoolConfig(
        provider="test",
        manager_name="test_balancer",
        strategy=RotationStrategy.ROUND_ROBIN,
        manager_class="test.Manager",
        storage_class="test.Storage",
        credentials=[
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="cred_a",
            ),
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="cred_b",
            ),
        ],
    )

    # Mock factory to return test managers
    factory = Mock(spec=AuthManagerFactory)
    factory.create_from_source = AsyncMock(
        side_effect=[
            MockAuthManager("token-a"),
            MockAuthManager("token-b"),
        ]
    )

    manager = await CredentialBalancerTokenManager.create(pool, factory=factory)

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
async def test_failover_after_failure() -> None:
    """Test that failover strategy switches to backup after failure."""
    pool = CredentialPoolConfig(
        provider="test",
        manager_name="test_failover",
        strategy=RotationStrategy.FAILOVER,
        manager_class="test.Manager",
        storage_class="test.Storage",
        credentials=[
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="primary",
            ),
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="backup",
            ),
        ],
        max_failures_before_disable=1,
    )

    factory = Mock(spec=AuthManagerFactory)
    factory.create_from_source = AsyncMock(
        side_effect=[
            MockAuthManager("token-primary"),
            MockAuthManager("token-backup"),
        ]
    )

    manager = await CredentialBalancerTokenManager.create(pool, factory=factory)

    token_first = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 401)

    token_second = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 200)

    assert token_first == "token-primary"
    assert token_second == "token-backup"


@pytest.mark.asyncio
async def test_all_credentials_exhausted() -> None:
    """Test that manager raises error when all credentials are exhausted."""
    pool = CredentialPoolConfig(
        provider="test",
        manager_name="test_exhausted",
        strategy=RotationStrategy.FAILOVER,
        manager_class="test.Manager",
        storage_class="test.Storage",
        credentials=[
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="only_cred",
            ),
        ],
        max_failures_before_disable=1,
        cooldown_seconds=9999,
    )

    factory = Mock(spec=AuthManagerFactory)
    factory.create_from_source = AsyncMock(return_value=MockAuthManager("token-only"))

    manager = await CredentialBalancerTokenManager.create(pool, factory=factory)

    token = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 401)

    with pytest.raises(
        AuthenticationError, match="No credential is currently available"
    ):
        await asyncio.wait_for(manager.get_access_token(), timeout=5)


@pytest.mark.asyncio
async def test_cooldown_recovery() -> None:
    """Test that credentials recover after cooldown period."""
    pool = CredentialPoolConfig(
        provider="test",
        manager_name="test_cooldown",
        strategy=RotationStrategy.FAILOVER,
        manager_class="test.Manager",
        storage_class="test.Storage",
        credentials=[
            CredentialManager(
                manager_class="test.Manager",
                storage_class="test.Storage",
                label="temp_fail",
            ),
        ],
        max_failures_before_disable=1,
        cooldown_seconds=0.1,
    )

    factory = Mock(spec=AuthManagerFactory)
    factory.create_from_source = AsyncMock(return_value=MockAuthManager("token-temp"))

    manager = await CredentialBalancerTokenManager.create(pool, factory=factory)

    token_first = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    request_id = _pop_request_id(manager)
    await manager.handle_response_event(request_id, 401)

    await asyncio.sleep(0.15)

    token_second = await asyncio.wait_for(manager.get_access_token(), timeout=5)
    assert token_first == "token-temp"
    assert token_second == "token-temp"
