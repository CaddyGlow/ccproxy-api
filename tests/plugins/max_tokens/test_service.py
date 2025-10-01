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
