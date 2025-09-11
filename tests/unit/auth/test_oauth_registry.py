"""Tests for OAuth provider registry."""

from typing import Any

import pytest

from ccproxy.auth.oauth.registry import (
    OAuthProviderInfo,
    OAuthRegistry,
)


class MockOAuthProvider:
    """Mock OAuth provider for testing."""

    def __init__(self, name: str = "test-provider"):
        self.provider_name = name
        self.provider_display_name = f"Test {name}"
        self.supports_pkce = True
        self.supports_refresh = True

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        return f"https://auth.example.com/authorize?state={state}"

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        return {"access_token": "test_token", "refresh_token": "test_refresh"}

    async def refresh_access_token(self, refresh_token: str) -> Any:
        return {"access_token": "new_token", "refresh_token": "new_refresh"}

    async def revoke_token(self, token: str) -> None:
        pass

    def get_provider_info(self) -> OAuthProviderInfo:
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="Test provider",
            supports_pkce=self.supports_pkce,
            scopes=["read", "write"],
            is_available=True,
            plugin_name="test-plugin",
        )

    def get_storage(self) -> Any:
        return None

    def get_credential_summary(self, credentials: Any) -> dict[str, Any]:
        return {
            "provider": self.provider_display_name,
            "authenticated": bool(credentials),
        }


@pytest.fixture
def registry():
    """Create a fresh registry for testing."""
    return OAuthRegistry()


@pytest.fixture
def mock_provider():
    """Create a mock OAuth provider."""
    return MockOAuthProvider()


class TestOAuthRegistry:
    """Test OAuth provider registry."""

    def test_register_provider(self, registry, mock_provider):
        """Test provider registration."""
        registry.register(mock_provider)

        providers = registry.list()
        assert "test-provider" in providers
        assert providers["test-provider"].display_name == "Test test-provider"

    def test_get_provider(self, registry, mock_provider):
        """Test getting a registered provider."""
        registry.register(mock_provider)

        provider = registry.get("test-provider")
        assert provider is not None
        assert provider.provider_name == "test-provider"

    def test_get_nonexistent_provider(self, registry):
        """Test getting a non-existent provider."""
        provider = registry.get("nonexistent")
        assert provider is None

    def test_unregister_provider(self, registry, mock_provider):
        """Test unregistering a provider."""
        registry.register(mock_provider)
        assert "test-provider" in registry.list()

        registry.unregister("test-provider")
        assert "test-provider" not in registry.list()

    def test_register_duplicate_provider(self, registry, mock_provider):
        """Test registering a duplicate provider raises an error."""
        registry.register(mock_provider)

        # Create a new provider with the same name
        new_provider = MockOAuthProvider("test-provider")
        new_provider.provider_display_name = "New Test Provider"

        # Should raise ValueError for duplicate registration
        with pytest.raises(ValueError, match="already registered"):
            registry.register(new_provider)

    def test_list_providers_empty(self, registry):
        """Test listing providers when registry is empty."""
        providers = registry.list()
        assert providers == {}

    def test_list_multiple_providers(self, registry):
        """Test listing multiple providers."""
        provider1 = MockOAuthProvider("provider1")
        provider2 = MockOAuthProvider("provider2")
        provider3 = MockOAuthProvider("provider3")

        registry.register(provider1)
        registry.register(provider2)
        registry.register(provider3)

        providers = registry.list()
        assert len(providers) == 3
        assert "provider1" in providers
        assert "provider2" in providers
        assert "provider3" in providers

    @pytest.mark.asyncio
    async def test_provider_authorization_url(self, registry, mock_provider):
        """Test getting authorization URL through registry."""
        registry.register(mock_provider)

        provider = registry.get("test-provider")
        assert provider is not None

        url = await provider.get_authorization_url("test_state", "test_verifier")
        assert "test_state" in url
        assert url.startswith("https://auth.example.com/authorize")

    @pytest.mark.asyncio
    async def test_provider_callback(self, registry, mock_provider):
        """Test handling callback through registry."""
        registry.register(mock_provider)

        provider = registry.get("test-provider")
        assert provider is not None

        result = await provider.handle_callback(
            "test_code", "test_state", "test_verifier"
        )
        assert result["access_token"] == "test_token"
        assert result["refresh_token"] == "test_refresh"
