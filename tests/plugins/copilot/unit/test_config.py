"""Unit tests for Copilot plugin configuration."""

from ccproxy.plugins.copilot.config import (
    CopilotConfig,
    CopilotOAuthConfig,
    CopilotProviderConfig,
)


class TestCopilotOAuthConfig:
    """Test cases for CopilotOAuthConfig."""

    def test_default_initialization(self) -> None:
        """Test default OAuth configuration."""
        config = CopilotOAuthConfig()

        assert config.client_id == "Iv1.b507a08c87ecfe98"
        assert config.authorize_url == "https://github.com/login/device/code"
        assert config.token_url == "https://github.com/login/oauth/access_token"
        assert (
            config.copilot_token_url
            == "https://api.github.com/copilot_internal/v2/token"
        )
        assert config.scopes == ["read:user"]
        assert config.use_pkce is True
        assert config.request_timeout == 30
        assert config.callback_timeout == 300
        assert config.callback_port == 8080

    def test_custom_initialization(self) -> None:
        """Test custom OAuth configuration."""
        config = CopilotOAuthConfig(
            client_id="custom-client-id",
            authorize_url="https://custom.example.com/device/code",
            token_url="https://custom.example.com/oauth/token",
            copilot_token_url="https://custom.example.com/copilot/token",
            scopes=["read:user", "copilot", "custom"],
            use_pkce=False,
            request_timeout=60,
            callback_timeout=600,
            callback_port=9000,
        )

        assert config.client_id == "custom-client-id"
        assert config.authorize_url == "https://custom.example.com/device/code"
        assert config.token_url == "https://custom.example.com/oauth/token"
        assert config.copilot_token_url == "https://custom.example.com/copilot/token"
        assert config.scopes == ["read:user", "copilot", "custom"]
        assert config.use_pkce is False
        assert config.request_timeout == 60
        assert config.callback_timeout == 600
        assert config.callback_port == 9000

    def test_get_redirect_uri_default(self) -> None:
        """Test redirect URI generation with default port."""
        config = CopilotOAuthConfig()
        assert config.get_redirect_uri() == "http://localhost:8080/callback"

    def test_get_redirect_uri_custom_port(self) -> None:
        """Test redirect URI generation with custom port."""
        config = CopilotOAuthConfig(callback_port=9000)
        assert config.get_redirect_uri() == "http://localhost:9000/callback"

    def test_get_redirect_uri_explicit(self) -> None:
        """Test explicit redirect URI."""
        config = CopilotOAuthConfig(redirect_uri="https://example.com/callback")
        assert config.get_redirect_uri() == "https://example.com/callback"

    def test_serialization(self) -> None:
        """Test configuration serialization."""
        config = CopilotOAuthConfig(
            client_id="test-client",
            scopes=["read:user", "copilot"],
            callback_port=9000,
        )

        data = config.model_dump()

        assert data["client_id"] == "test-client"
        assert data["scopes"] == ["read:user", "copilot"]
        assert data["callback_port"] == 9000
        assert data["use_pkce"] is True

    def test_deserialization(self) -> None:
        """Test configuration deserialization."""
        data = {
            "client_id": "test-client",
            "authorize_url": "https://example.com/auth",
            "token_url": "https://example.com/token",
            "copilot_token_url": "https://example.com/copilot",
            "scopes": ["read:user", "admin"],
            "use_pkce": False,
            "request_timeout": 60,
        }

        config = CopilotOAuthConfig(**data)

        assert config.client_id == "test-client"
        assert config.authorize_url == "https://example.com/auth"
        assert config.token_url == "https://example.com/token"
        assert config.copilot_token_url == "https://example.com/copilot"
        assert config.scopes == ["read:user", "admin"]
        assert config.use_pkce is False
        assert config.request_timeout == 60


class TestCopilotProviderConfig:
    """Test cases for CopilotProviderConfig."""

    def test_default_initialization(self) -> None:
        """Test default provider configuration."""
        config = CopilotProviderConfig()

        assert config.account_type == "individual"
        assert config.base_url is None
        assert config.request_timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    def test_custom_initialization(self) -> None:
        """Test custom provider configuration."""
        config = CopilotProviderConfig(
            account_type="business",
            base_url="https://custom.example.com",
            request_timeout=60,
            max_retries=5,
            retry_delay=2.0,
        )

        assert config.account_type == "business"
        assert config.base_url == "https://custom.example.com"
        assert config.request_timeout == 60
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    def test_get_base_url_individual(self) -> None:
        """Test base URL generation for individual account."""
        config = CopilotProviderConfig(account_type="individual")
        assert config.get_base_url() == "https://api.githubcopilot.com"

    def test_get_base_url_business(self) -> None:
        """Test base URL generation for business account."""
        config = CopilotProviderConfig(account_type="business")
        assert config.get_base_url() == "https://api.business.githubcopilot.com"

    def test_get_base_url_enterprise(self) -> None:
        """Test base URL generation for enterprise account."""
        config = CopilotProviderConfig(account_type="enterprise")
        assert config.get_base_url() == "https://api.enterprise.githubcopilot.com"

    def test_get_base_url_explicit(self) -> None:
        """Test explicit base URL."""
        config = CopilotProviderConfig(
            account_type="business",
            base_url="https://custom.example.com",
        )
        assert config.get_base_url() == "https://custom.example.com"

    def test_get_base_url_unknown_account_type(self) -> None:
        """Test base URL fallback for unknown account type."""
        config = CopilotProviderConfig(account_type="unknown")
        assert config.get_base_url() == "https://api.githubcopilot.com"


class TestCopilotConfig:
    """Test cases for CopilotConfig."""

    def test_default_initialization(self) -> None:
        """Test default Copilot configuration."""
        config = CopilotConfig()

        assert config.enabled is True
        assert isinstance(config.oauth, CopilotOAuthConfig)
        assert isinstance(config.provider, CopilotProviderConfig)
        assert config.oauth.client_id == "Iv1.b507a08c87ecfe98"
        assert config.provider.account_type == "individual"
        assert "Content-Type" in config.api_headers
        assert config.api_headers["Content-Type"] == "application/json"

    def test_custom_oauth_config(self) -> None:
        """Test Copilot configuration with custom OAuth config."""
        oauth_config = CopilotOAuthConfig(
            client_id="custom-client",
            scopes=["read:user", "copilot", "admin"],
        )

        config = CopilotConfig(oauth=oauth_config)

        assert config.oauth is oauth_config
        assert config.oauth.client_id == "custom-client"
        assert config.oauth.scopes == ["read:user", "copilot", "admin"]

    def test_custom_provider_config(self) -> None:
        """Test Copilot configuration with custom provider config."""
        provider_config = CopilotProviderConfig(
            account_type="business",
            request_timeout=60,
        )

        config = CopilotConfig(provider=provider_config)

        assert config.provider is provider_config
        assert config.provider.account_type == "business"
        assert config.provider.request_timeout == 60

    def test_serialization(self) -> None:
        """Test configuration serialization."""
        oauth_config = CopilotOAuthConfig(
            client_id="test-client",
        )
        provider_config = CopilotProviderConfig(
            account_type="enterprise",
        )
        config = CopilotConfig(oauth=oauth_config, provider=provider_config)

        data = config.model_dump()

        assert "oauth" in data
        assert "provider" in data
        assert data["oauth"]["client_id"] == "test-client"
        assert data["provider"]["account_type"] == "enterprise"

    def test_deserialization(self) -> None:
        """Test configuration deserialization."""
        data = {
            "enabled": False,
            "oauth": {
                "client_id": "test-client",
                "scopes": ["read:user", "copilot"],
                "use_pkce": False,
            },
            "provider": {
                "account_type": "business",
                "request_timeout": 60,
            },
        }

        config = CopilotConfig(**data)

        assert config.enabled is False
        assert isinstance(config.oauth, CopilotOAuthConfig)
        assert isinstance(config.provider, CopilotProviderConfig)
        assert config.oauth.client_id == "test-client"
        assert config.oauth.scopes == ["read:user", "copilot"]
        assert config.oauth.use_pkce is False
        assert config.provider.account_type == "business"
        assert config.provider.request_timeout == 60

    def test_nested_config_update(self) -> None:
        """Test updating nested configuration."""
        config = CopilotConfig()

        # Verify default
        assert config.provider.account_type == "individual"

        # Update with new config
        new_provider = CopilotProviderConfig(
            account_type="business",
            request_timeout=60,
        )
        config.provider = new_provider

        assert config.provider.account_type == "business"
        assert config.provider.request_timeout == 60

    def test_validation_preserves_defaults(self) -> None:
        """Test that validation preserves default values."""
        # Create config with partial data
        data = {
            "oauth": {
                "client_id": "custom-client",
            },
            "provider": {
                "account_type": "business",
            },
        }

        config = CopilotConfig(**data)

        # Should preserve defaults for unspecified fields
        assert config.oauth.client_id == "custom-client"
        assert config.oauth.use_pkce is True  # Default preserved
        assert config.oauth.scopes == ["read:user"]  # Default preserved
        assert config.provider.account_type == "business"
        assert config.provider.request_timeout == 30  # Default preserved

    def test_config_copy_behavior(self) -> None:
        """Test configuration copy behavior."""
        original = CopilotConfig()
        original.oauth = CopilotOAuthConfig(
            client_id="original-client",
        )
        original.provider = CopilotProviderConfig(
            account_type="individual",
        )

        # Create copy through model validation
        copy_data = original.model_dump()
        copy = CopilotConfig(**copy_data)

        # Should have same values
        assert copy.oauth.client_id == original.oauth.client_id
        assert copy.provider.account_type == original.provider.account_type

        # But should be independent objects
        copy.oauth = CopilotOAuthConfig(
            client_id="modified-client",
        )

        # Original should be unchanged
        assert original.oauth.client_id == "original-client"

    def test_api_headers_customization(self) -> None:
        """Test API headers customization."""
        custom_headers = {
            "Content-Type": "application/json",
            "Custom-Header": "custom-value",
        }

        config = CopilotConfig(api_headers=custom_headers)

        assert config.api_headers == custom_headers
        assert config.api_headers["Custom-Header"] == "custom-value"

    def test_disabled_config(self) -> None:
        """Test disabled plugin configuration."""
        config = CopilotConfig(enabled=False)

        assert config.enabled is False
        # Other defaults should still be set
        assert isinstance(config.oauth, CopilotOAuthConfig)
        assert isinstance(config.provider, CopilotProviderConfig)
