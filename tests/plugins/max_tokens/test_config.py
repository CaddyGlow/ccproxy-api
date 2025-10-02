"""Tests for max_tokens configuration."""

from ccproxy.plugins.max_tokens.config import MaxTokensConfig


class TestMaxTokensConfig:
    """Test cases for MaxTokensConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MaxTokensConfig()

        assert config.enabled is True
        assert config.fallback_max_tokens == 4096
        assert config.apply_to_all_providers is True
        assert "claude_api" in config.target_providers
        assert "claude_sdk" in config.target_providers
        assert config.require_pricing_data is False
        assert config.log_modifications is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MaxTokensConfig(
            enabled=False,
            fallback_max_tokens=2048,
            apply_to_all_providers=False,
            target_providers=["claude_api"],
            require_pricing_data=True,
            log_modifications=False,
        )

        assert config.enabled is False
        assert config.fallback_max_tokens == 2048
        assert config.apply_to_all_providers is False
        assert config.target_providers == ["claude_api"]
        assert config.require_pricing_data is True
        assert config.log_modifications is False

    def test_should_process_provider_all_providers(self) -> None:
        """Test provider processing when apply_to_all_providers is True."""
        config = MaxTokensConfig(apply_to_all_providers=True)

        assert config.should_process_provider("claude_api") is True
        assert config.should_process_provider("unknown_provider") is True
        assert config.should_process_provider("any_provider") is True

    def test_should_process_provider_specific(self) -> None:
        """Test provider processing when apply_to_all_providers is False."""
        config = MaxTokensConfig(
            apply_to_all_providers=False,
            target_providers=["claude_api", "claude_sdk"],
        )

        assert config.should_process_provider("claude_api") is True
        assert config.should_process_provider("claude_sdk") is True
        assert config.should_process_provider("copilot") is False
        assert config.should_process_provider("unknown_provider") is False

    def test_get_modification_reason(self) -> None:
        """Test getting modification reasons."""
        config = MaxTokensConfig()

        assert "missing" in config.get_modification_reason("missing")
        assert "invalid" in config.get_modification_reason("invalid")
        assert "exceeded" in config.get_modification_reason("exceeded")
        assert "Unknown reason" in config.get_modification_reason("unknown_type")

    def test_config_validation_string_target_providers(self) -> None:
        """Test configuration validation when target_providers is a string."""
        # This tests the model validator that converts string to list
        config = MaxTokensConfig(target_providers="claude_api")

        assert isinstance(config.target_providers, list)
        assert config.target_providers == ["claude_api"]

    def test_enforce_mode_default(self) -> None:
        """Test that enforce_mode defaults to False."""
        config = MaxTokensConfig()
        assert config.enforce_mode is False

    def test_enforce_mode_enabled(self) -> None:
        """Test enforce_mode configuration."""
        config = MaxTokensConfig(enforce_mode=True)
        assert config.enforce_mode is True

    def test_modification_reasons_includes_enforced(self) -> None:
        """Test that modification reasons include enforce mode."""
        config = MaxTokensConfig()
        assert "enforced" in config.modification_reasons
        assert (
            config.modification_reasons["enforced"]
            == "max_tokens enforced to model limit (enforce mode)"
        )

    def test_prioritize_local_file_default(self) -> None:
        """Test that prioritize_local_file defaults to False."""
        config = MaxTokensConfig()
        assert config.prioritize_local_file is False

    def test_prioritize_local_file_enabled(self) -> None:
        """Test prioritize_local_file configuration."""
        config = MaxTokensConfig(prioritize_local_file=True)
        assert config.prioritize_local_file is True
