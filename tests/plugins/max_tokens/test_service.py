"""Tests for max_tokens service."""

import pytest

from ccproxy.plugins.max_tokens.config import MaxTokensConfig
from ccproxy.plugins.max_tokens.service import TokenLimitsService


class TestTokenLimitsService:
    """Test cases for TokenLimitsService."""

    @pytest.fixture
    def config(self) -> MaxTokensConfig:
        """Create test configuration."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
        )

    @pytest.fixture
    def service(self, config: MaxTokensConfig) -> TokenLimitsService:
        """Create token limits service instance."""
        service = TokenLimitsService(config)
        service.initialize()
        return service

    def test_get_max_output_tokens_known_model(
        self, service: TokenLimitsService
    ) -> None:
        """Test getting max output tokens for known models."""
        # Test Claude 3.5 Sonnet (loads from pricing cache if available)
        max_tokens = service.get_max_output_tokens("claude-3-5-sonnet-20241022")
        assert max_tokens == 8192

        # Test Claude 3 Opus
        max_tokens = service.get_max_output_tokens("claude-3-opus-20240229")
        assert max_tokens == 4096

        # Test Claude 3 Haiku
        max_tokens = service.get_max_output_tokens("claude-3-haiku-20240307")
        assert max_tokens == 4096

    def test_get_max_output_tokens_variant_models(
        self, service: TokenLimitsService
    ) -> None:
        """Test variant models from pricing cache."""
        # The pricing cache includes many model variants
        # We just verify that models in the cache can be retrieved
        assert len(service.token_limits_data.models) > 0

    def test_get_max_output_tokens_unknown_model(
        self, service: TokenLimitsService
    ) -> None:
        """Test getting max output tokens for unknown model."""
        max_tokens = service.get_max_output_tokens("unknown-model")
        assert max_tokens is None

    def test_should_modify_missing_max_tokens(
        self, service: TokenLimitsService
    ) -> None:
        """Test modification when max_tokens is missing."""
        request_data = {"model": "claude-3-5-sonnet-20241022"}
        should_modify, reason = service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is True
        assert reason == "missing"

    def test_should_modify_invalid_max_tokens(
        self, service: TokenLimitsService
    ) -> None:
        """Test modification when max_tokens is invalid."""
        # Test non-integer max_tokens
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": "invalid"}
        should_modify, reason = service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is True
        assert reason == "invalid"

        # Test negative max_tokens
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": -1}
        should_modify, reason = service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is True
        assert reason == "invalid"

    def test_should_modify_exceeded_max_tokens(
        self, service: TokenLimitsService
    ) -> None:
        """Test modification when max_tokens exceeds model limit."""
        request_data = {"model": "claude-3-opus-20240229", "max_tokens": 5000}
        should_modify, reason = service.should_modify_max_tokens(
            request_data, "claude-3-opus-20240229"
        )

        assert should_modify is True
        assert reason == "exceeded"

    def test_should_not_modify_valid_max_tokens(
        self, service: TokenLimitsService
    ) -> None:
        """Test no modification when max_tokens is valid."""
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000}
        should_modify, reason = service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is False
        assert reason == "none"

    def test_modify_max_tokens_missing(self, service: TokenLimitsService) -> None:
        """Test modifying request with missing max_tokens."""
        request_data = {"model": "claude-3-5-sonnet-20241022"}
        modified_data, modification = service.modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.original_max_tokens is None
        assert modification.new_max_tokens == 8192  # From pricing cache
        assert modification.reason == "max_tokens was missing from request"
        assert modified_data["max_tokens"] == 8192

    def test_modify_max_tokens_exceeded(self, service: TokenLimitsService) -> None:
        """Test modifying request with exceeded max_tokens."""
        request_data = {"model": "claude-3-opus-20240229", "max_tokens": 5000}
        modified_data, modification = service.modify_max_tokens(
            request_data, "claude-3-opus-20240229"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.original_max_tokens == 5000
        assert modification.new_max_tokens == 4096
        assert modification.reason == "max_tokens exceeded model limit"
        assert modified_data["max_tokens"] == 4096

    def test_modify_max_tokens_unknown_model_fallback(
        self, service: TokenLimitsService
    ) -> None:
        """Test modifying request for unknown model using fallback."""
        request_data = {"model": "unknown-model"}
        modified_data, modification = service.modify_max_tokens(
            request_data, "unknown-model"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.original_max_tokens is None
        assert modification.new_max_tokens == 2048  # fallback value
        assert "max_tokens was missing from request" in modification.reason
        assert modified_data["max_tokens"] == 2048

    def test_no_modification_needed(self, service: TokenLimitsService) -> None:
        """Test no modification when max_tokens is already valid."""
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000}
        modified_data, modification = service.modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert modification is None
        assert modified_data == request_data


class TestTokenLimitsServiceLocalFilePriority:
    """Test cases for local file priority configuration."""

    @pytest.fixture
    def prioritize_config(self) -> MaxTokensConfig:
        """Create configuration with local file prioritized."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
            prioritize_local_file=True,
        )

    @pytest.fixture
    def fallback_config(self) -> MaxTokensConfig:
        """Create configuration with local file as fallback (default)."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
            prioritize_local_file=False,  # Default behavior
        )

    @pytest.fixture
    def prioritize_service(
        self, prioritize_config: MaxTokensConfig
    ) -> TokenLimitsService:
        """Create service with local file prioritized."""
        service = TokenLimitsService(prioritize_config)
        service.initialize()
        return service

    @pytest.fixture
    def fallback_service(self, fallback_config: MaxTokensConfig) -> TokenLimitsService:
        """Create service with local file as fallback."""
        service = TokenLimitsService(fallback_config)
        service.initialize()
        return service

    def test_local_file_as_fallback_only_adds_missing_models(
        self, fallback_service: TokenLimitsService
    ) -> None:
        """Test that fallback mode only adds models not found in pricing cache."""
        # The local file has gpt-5 and claude-opus-4-1-20250805
        # Since pricing cache already has these models, local file values should NOT be used

        # Check a model that exists in both sources - should use pricing cache value
        pricing_cache_value = fallback_service.get_max_output_tokens(
            "claude-opus-4-1-20250805"
        )
        assert (
            pricing_cache_value == 32000
        )  # This should be from pricing cache, not local file

        # Check gpt-5 - should use pricing cache value (128000)
        gpt5_value = fallback_service.get_max_output_tokens("gpt-5")
        assert gpt5_value == 128000  # From pricing cache

    def test_prioritize_local_file_overrides_pricing_cache(
        self, prioritize_service: TokenLimitsService
    ) -> None:
        """Test that prioritize mode uses local file values over pricing cache."""
        # When prioritized, local file values should override pricing cache

        # claude-opus-4-1-20250805 should use local file value (32000)
        local_file_value = prioritize_service.get_max_output_tokens(
            "claude-opus-4-1-20250805"
        )
        assert local_file_value == 32000  # From local file

        # gpt-5 should use local file value (128000)
        gpt5_value = prioritize_service.get_max_output_tokens("gpt-5")
        assert gpt5_value == 128000  # From local file


class TestTokenLimitsServiceEnforceMode:
    """Test cases for TokenLimitsService with enforce mode enabled."""

    @pytest.fixture
    def enforce_config(self) -> MaxTokensConfig:
        """Create test configuration with enforce mode enabled."""
        return MaxTokensConfig(
            enabled=True,
            fallback_max_tokens=2048,
            log_modifications=False,
            enforce_mode=True,
        )

    @pytest.fixture
    def enforce_service(self, enforce_config: MaxTokensConfig) -> TokenLimitsService:
        """Create token limits service instance with enforce mode."""
        service = TokenLimitsService(enforce_config)
        service.initialize()
        return service

    def test_should_modify_always_in_enforce_mode(
        self, enforce_service: TokenLimitsService
    ) -> None:
        """Test that enforce mode always modifies regardless of current max_tokens."""
        # Test with valid max_tokens (normally wouldn't modify)
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000}
        should_modify, reason = enforce_service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is True
        assert reason == "enforced"

        # Test with missing max_tokens
        request_data = {"model": "claude-3-5-sonnet-20241022"}
        should_modify, reason = enforce_service.should_modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert should_modify is True
        assert reason == "enforced"

        # Test with exceeded max_tokens
        request_data = {"model": "claude-3-opus-20240229", "max_tokens": 5000}
        should_modify, reason = enforce_service.should_modify_max_tokens(
            request_data, "claude-3-opus-20240229"
        )

        assert should_modify is True
        assert reason == "enforced"

    def test_modify_max_tokens_always_enforces_to_model_limit(
        self, enforce_service: TokenLimitsService
    ) -> None:
        """Test that enforce mode always sets max_tokens to model limit."""
        # Test with existing valid max_tokens
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 1000}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.original_max_tokens == 1000
        assert modification.new_max_tokens == 8192  # Model limit
        assert (
            modification.reason == "max_tokens enforced to model limit (enforce mode)"
        )
        assert modified_data["max_tokens"] == 8192

        # Test with max_tokens already at model limit (still creates modification entry but no actual change)
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 8192}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022"
        )

        assert modification is not None
        # When original and new values are the same, was_modified() returns False
        assert modification.was_modified() is False
        assert modification.original_max_tokens == 8192
        assert modification.new_max_tokens == 8192  # Same value
        assert (
            modification.reason == "max_tokens enforced to model limit (enforce mode)"
        )

    def test_modify_max_tokens_enforce_mode_with_fallback(
        self, enforce_service: TokenLimitsService
    ) -> None:
        """Test enforce mode with unknown model using fallback."""
        request_data = {"model": "unknown-model", "max_tokens": 5000}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "unknown-model"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.original_max_tokens == 5000
        assert modification.new_max_tokens == 2048  # fallback value
        assert (
            modification.reason == "max_tokens enforced to model limit (enforce mode)"
        )
        assert modified_data["max_tokens"] == 2048

    def test_enforce_mode_respects_provider_selection(
        self, enforce_service: TokenLimitsService
    ) -> None:
        """Test that enforce mode behavior is independent of provider selection."""
        # The service itself doesn't handle provider selection - that's done at the adapter level
        # This test verifies that the service always modifies in enforce mode regardless of provider
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 100}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "claude-3-5-sonnet-20241022", "claude_api"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.new_max_tokens == 8192

    def test_enforce_mode_uses_local_token_limits_file(
        self, enforce_service: TokenLimitsService
    ) -> None:
        """Test that enforce mode uses values from local token_limits.json file."""
        # Test with a model that has specific limits in the local file
        request_data = {"model": "claude-opus-4-1-20250805", "max_tokens": 1000}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "claude-opus-4-1-20250805"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.new_max_tokens == 32000  # From local token_limits.json

        # Test with gpt-5 model
        request_data = {"model": "gpt-5", "max_tokens": 5000}
        modified_data, modification = enforce_service.modify_max_tokens(
            request_data, "gpt-5"
        )

        assert modification is not None
        assert modification.was_modified() is True
        assert modification.new_max_tokens == 128000  # From local token_limits.json
