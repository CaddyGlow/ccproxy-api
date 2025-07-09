"""Tests for cost calculator functionality."""

from unittest.mock import Mock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.exceptions import ValidationError
from ccproxy.metrics.calculator import CostCalculator, ModelPricing, get_cost_calculator


class TestModelPricing:
    """Test the ModelPricing class."""

    def test_model_pricing_creation(self):
        """Test creating a ModelPricing instance."""
        pricing = ModelPricing(
            input_tokens_per_1k=3.0,
            output_tokens_per_1k=15.0,
            cache_creation_tokens_per_1k=3.75,
            cache_read_tokens_per_1k=0.30,
        )

        assert pricing.input_tokens_per_1k == 3.0
        assert pricing.output_tokens_per_1k == 15.0
        assert pricing.cache_creation_tokens_per_1k == 3.75
        assert pricing.cache_read_tokens_per_1k == 0.30

    def test_model_pricing_defaults(self):
        """Test default values for optional fields."""
        pricing = ModelPricing(
            input_tokens_per_1k=3.0,
            output_tokens_per_1k=15.0,
        )

        assert pricing.input_tokens_per_1k == 3.0
        assert pricing.output_tokens_per_1k == 15.0
        assert pricing.cache_creation_tokens_per_1k == 0.0
        assert pricing.cache_read_tokens_per_1k == 0.0

    def test_model_pricing_validation(self):
        """Test that negative values are rejected."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelPricing(
                input_tokens_per_1k=-1.0,
                output_tokens_per_1k=15.0,
            )


class TestCostCalculator:
    """Test the CostCalculator class."""

    def test_normalize_model_name(self):
        """Test model name normalization."""
        calculator = CostCalculator()

        # Test Claude model normalization
        assert (
            calculator._normalize_model_name("claude-3-5-sonnet")
            == "claude-3-5-sonnet-20241022"
        )
        assert (
            calculator._normalize_model_name("claude-3-opus")
            == "claude-3-opus-20240229"
        )
        assert (
            calculator._normalize_model_name("claude-3-haiku")
            == "claude-3-haiku-20240307"
        )

        # Test OpenAI model mapping
        assert calculator._normalize_model_name("gpt-4") == "claude-3-sonnet-20240229"
        assert (
            calculator._normalize_model_name("gpt-4-turbo")
            == "claude-3-5-sonnet-20241022"
        )
        assert (
            calculator._normalize_model_name("gpt-3.5-turbo")
            == "claude-3-haiku-20240307"
        )

        # Test exact model names are preserved
        assert (
            calculator._normalize_model_name("claude-3-5-sonnet-20241022")
            == "claude-3-5-sonnet-20241022"
        )

    def test_get_model_pricing_default(self):
        """Test getting default pricing for a model."""
        calculator = CostCalculator()

        pricing = calculator.get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing.input_tokens_per_1k == 3.0
        assert pricing.output_tokens_per_1k == 15.0
        assert pricing.cache_creation_tokens_per_1k == 3.75
        assert pricing.cache_read_tokens_per_1k == 0.30

    def test_get_model_pricing_fallback(self):
        """Test fallback to default pricing for unknown models."""
        calculator = CostCalculator()

        pricing = calculator.get_model_pricing("unknown-model")
        assert pricing.input_tokens_per_1k == 5.0
        assert pricing.output_tokens_per_1k == 15.0
        assert pricing.cache_creation_tokens_per_1k == 6.25
        assert pricing.cache_read_tokens_per_1k == 0.50

    @patch("ccproxy.metrics.calculator.get_settings")
    def test_get_model_pricing_from_settings(self, mock_get_settings):
        """Test getting pricing from settings configuration."""
        mock_settings = Mock()
        mock_settings.model_pricing = {
            "custom-model": {
                "input": 2.0,
                "output": 10.0,
                "cache_creation": 2.5,
                "cache_read": 0.25,
            }
        }
        mock_get_settings.return_value = mock_settings

        calculator = CostCalculator()
        pricing = calculator.get_model_pricing("custom-model")

        assert pricing.input_tokens_per_1k == 2.0
        assert pricing.output_tokens_per_1k == 10.0
        assert pricing.cache_creation_tokens_per_1k == 2.5
        assert pricing.cache_read_tokens_per_1k == 0.25

    def test_calculate_cost_basic(self):
        """Test basic cost calculation."""
        calculator = CostCalculator()

        # Test with claude-3-5-sonnet pricing (3.0/1K input, 15.0/1K output)
        cost = calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
        )

        expected_cost = (1000 / 1000) * 3.0 + (500 / 1000) * 15.0  # 3.0 + 7.5 = 10.5
        assert cost == expected_cost

    def test_calculate_cost_with_cache(self):
        """Test cost calculation with cache tokens."""
        calculator = CostCalculator()

        cost = calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_creation_tokens=200,
            cache_read_tokens=100,
        )

        # Expected: 3.0 + 7.5 + 0.75 + 0.03 = 11.28
        expected_cost = (
            (1000 / 1000) * 3.0  # input
            + (500 / 1000) * 15.0  # output
            + (200 / 1000) * 3.75  # cache creation
            + (100 / 1000) * 0.30  # cache read
        )
        assert cost == expected_cost

    def test_calculate_cost_validation(self):
        """Test that negative token counts are rejected."""
        calculator = CostCalculator()

        with pytest.raises(ValidationError):
            calculator.calculate_cost(
                model="claude-3-5-sonnet-20241022",
                input_tokens=-1,
                output_tokens=500,
            )

        with pytest.raises(ValidationError):
            calculator.calculate_cost(
                model="claude-3-5-sonnet-20241022",
                input_tokens=1000,
                output_tokens=-1,
            )

    def test_estimate_streaming_cost(self):
        """Test streaming cost estimation."""
        calculator = CostCalculator()

        # Test without output token estimation
        cost = calculator.estimate_streaming_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
        )

        expected_cost = (1000 / 1000) * 3.0  # Only input cost
        assert cost == expected_cost

        # Test with output token estimation
        cost = calculator.estimate_streaming_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            estimated_output_tokens=500,
        )

        expected_cost = (1000 / 1000) * 3.0 + (500 / 1000) * 15.0
        assert cost == expected_cost


class TestGlobalCostCalculator:
    """Test the global cost calculator instance."""

    def test_get_cost_calculator_singleton(self):
        """Test that get_cost_calculator returns the same instance."""
        calculator1 = get_cost_calculator()
        calculator2 = get_cost_calculator()

        assert calculator1 is calculator2
        assert isinstance(calculator1, CostCalculator)
