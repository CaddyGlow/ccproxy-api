"""Tests for FastAPI factory pattern implementation.

This module tests the new factory-based approach to creating FastAPI
applications and clients with different configurations.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient

from ccproxy.config.settings import Settings
from tests.factories import FastAPIAppFactory, FastAPIClientFactory


@pytest.mark.unit
def test_fastapi_app_factory_basic(test_settings: Settings) -> None:
    """Test creating a basic FastAPI app using the factory."""
    factory = FastAPIAppFactory(default_settings=test_settings)
    app = factory.create_app()

    assert isinstance(app, FastAPI)
    assert app.title == "Claude Code Proxy API Server"


@pytest.mark.unit
def test_fastapi_app_factory_with_mock_claude(
    test_settings: Settings, mock_claude_service: AsyncMock
) -> None:
    """Test creating a FastAPI app with mocked Claude service."""
    factory = FastAPIAppFactory(default_settings=test_settings)
    app = factory.create_app(claude_service_mock=mock_claude_service)

    assert isinstance(app, FastAPI)
    # Check that dependency overrides were applied
    assert len(app.dependency_overrides) > 0


@pytest.mark.unit
def test_fastapi_app_factory_with_auth(
    test_settings: Settings, auth_settings: Settings
) -> None:
    """Test creating a FastAPI app with authentication enabled."""
    factory = FastAPIAppFactory(default_settings=test_settings)
    app = factory.create_app(settings=auth_settings, auth_enabled=True)

    assert isinstance(app, FastAPI)
    # Check that dependency overrides were applied
    assert len(app.dependency_overrides) > 0


@pytest.mark.unit
def test_fastapi_app_factory_composition(
    test_settings: Settings,
    auth_settings: Settings,
    mock_claude_service: AsyncMock,
) -> None:
    """Test creating a FastAPI app with multiple configurations composed."""
    factory = FastAPIAppFactory(default_settings=test_settings)
    app = factory.create_app(
        settings=auth_settings,
        claude_service_mock=mock_claude_service,
        auth_enabled=True,
    )

    assert isinstance(app, FastAPI)
    # Check that dependency overrides were applied for both auth and mock service
    assert len(app.dependency_overrides) >= 2


@pytest.mark.unit
def test_fastapi_client_factory_basic(test_settings: Settings) -> None:
    """Test creating a basic test client using the factory."""
    app_factory = FastAPIAppFactory(default_settings=test_settings)
    client_factory = FastAPIClientFactory(app_factory)

    client = client_factory.create_client()

    assert isinstance(client, TestClient)

    # Test that the client works
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_fastapi_client_factory_with_mock(
    test_settings: Settings, mock_claude_service: AsyncMock
) -> None:
    """Test creating a test client with mocked Claude service."""
    app_factory = FastAPIAppFactory(default_settings=test_settings)
    client_factory = FastAPIClientFactory(app_factory)

    client = client_factory.create_client(claude_service_mock=mock_claude_service)

    assert isinstance(client, TestClient)

    # Test that the client works
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fastapi_client_factory_async(test_settings: Settings) -> None:
    """Test creating an async test client using the factory."""
    app_factory = FastAPIAppFactory(default_settings=test_settings)
    client_factory = FastAPIClientFactory(app_factory)

    async with client_factory.create_async_client() as client:
        assert isinstance(client, AsyncClient)

        # Test that the async client works
        response = await client.get("/health")
        assert response.status_code == 200


@pytest.mark.unit
def test_factory_fixtures_integration(
    fastapi_app_factory: FastAPIAppFactory,
    fastapi_client_factory: FastAPIClientFactory,
    mock_claude_service: AsyncMock,
) -> None:
    """Test that the new factory fixtures work together correctly."""
    # Test app factory fixture
    app = fastapi_app_factory.create_app(claude_service_mock=mock_claude_service)
    assert isinstance(app, FastAPI)

    # Test client factory fixture
    client = fastapi_client_factory.create_client(
        claude_service_mock=mock_claude_service
    )
    assert isinstance(client, TestClient)

    # Test that the client works
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_backward_compatibility_legacy_fixtures(
    app_factory_basic: FastAPI,
    client_factory_basic: TestClient,
) -> None:
    """Test that legacy fixtures still work for backward compatibility."""
    # Test legacy app fixture
    assert isinstance(app_factory_basic, FastAPI)

    # Test legacy client fixture
    assert isinstance(client_factory_basic, TestClient)

    # Test that the client works
    response = client_factory_basic.get("/health")
    assert response.status_code == 200


@pytest.mark.unit
def test_factory_error_handling(test_settings: Settings) -> None:
    """Test that factory properly handles error cases."""
    # Test creating factory without default settings
    factory = FastAPIAppFactory()

    # Should raise error when no settings provided
    with pytest.raises(ValueError, match="Settings must be provided"):
        factory.create_app()


@pytest.mark.unit
def test_custom_dependency_overrides(test_settings: Settings) -> None:
    """Test that custom dependency overrides work correctly."""
    from ccproxy.config.settings import get_settings

    # Create a custom override
    def custom_override():
        return test_settings

    factory = FastAPIAppFactory(default_settings=test_settings)
    app = factory.create_app(dependency_overrides={get_settings: custom_override})

    assert isinstance(app, FastAPI)
    # Check that our custom override is in the app's overrides
    assert get_settings in app.dependency_overrides
