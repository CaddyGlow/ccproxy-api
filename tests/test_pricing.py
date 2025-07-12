"""Tests for the dynamic pricing system.

File: tests/test_pricing.py

This module tests the pricing infrastructure including cache management,
LiteLLM data loading, format conversion, and cost calculation with dynamic pricing.
Tests use real internal components and only mock external HTTP calls.
"""

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pytest_httpx import HTTPXMock

from ccproxy.metrics.calculator import CostCalculator
from ccproxy.pricing.cache import PricingCache
from ccproxy.pricing.loader import PricingLoader
from ccproxy.pricing.updater import PricingUpdater


@pytest.mark.unit
class TestPricingCache:
    """Test pricing cache management functionality."""

    def test_pricing_cache_initialization(self, tmp_path: Path) -> None:
        """Test pricing cache initialization with custom directory."""
        cache_dir = tmp_path / "test_cache"
        cache = PricingCache(cache_dir=str(cache_dir))

        assert cache.cache_dir == cache_dir
        assert cache.cache_file == cache_dir / "model_pricing.json"
        assert cache.cache_ttl_hours == 24
        assert (
            cache.source_url
            == "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
        )

    def test_pricing_cache_xdg_cache_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test pricing cache uses XDG_CACHE_HOME when available."""
        xdg_cache = tmp_path / "xdg_cache"
        xdg_cache.mkdir()
        monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))

        cache = PricingCache()
        assert cache.cache_dir == xdg_cache / "ccproxy"

    def test_cache_validity_check_file_not_exists(self, tmp_path: Path) -> None:
        """Test cache validity when file doesn't exist."""
        cache = PricingCache(cache_dir=str(tmp_path))
        assert not cache.is_cache_valid()

    def test_cache_validity_check_fresh_file(self, tmp_path: Path) -> None:
        """Test cache validity with fresh file."""
        cache = PricingCache(cache_dir=str(tmp_path), cache_ttl_hours=1)

        # Create fresh cache file
        cache.cache_file.write_text('{"test": "data"}')

        assert cache.is_cache_valid()

    def test_cache_validity_check_expired_file(self, tmp_path: Path) -> None:
        """Test cache validity with expired file."""
        cache = PricingCache(cache_dir=str(tmp_path), cache_ttl_hours=1)

        # Create cache file
        cache.cache_file.write_text('{"test": "data"}')

        # Modify timestamp to make it appear old
        import os

        old_time = time.time() - (2 * 3600)  # 2 hours ago
        os.utime(cache.cache_file, (old_time, old_time))

        assert not cache.is_cache_valid()

    def test_load_cached_data_success(self, tmp_path: Path) -> None:
        """Test successful loading of cached data."""
        cache = PricingCache(cache_dir=str(tmp_path))
        test_data = {"claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0}}

        cache.cache_file.write_text(json.dumps(test_data))

        loaded_data = cache.load_cached_data()
        assert loaded_data == test_data

    def test_load_cached_data_invalid_file(self, tmp_path: Path) -> None:
        """Test loading cached data with invalid JSON."""
        cache = PricingCache(cache_dir=str(tmp_path))

        cache.cache_file.write_text("invalid json")

        loaded_data = cache.load_cached_data()
        assert loaded_data is None

    async def test_download_pricing_data_success(self, httpx_mock: HTTPXMock) -> None:
        """Test successful pricing data download."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache()
        downloaded_data = await cache.download_pricing_data()

        assert downloaded_data == test_data

    async def test_download_pricing_data_failure(self, httpx_mock: HTTPXMock) -> None:
        """Test pricing data download failure."""
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            status_code=404,
        )

        cache = PricingCache()
        downloaded_data = await cache.download_pricing_data()

        assert downloaded_data is None

    def test_save_to_cache_success(self, tmp_path: Path) -> None:
        """Test successful saving to cache."""
        cache = PricingCache(cache_dir=str(tmp_path))
        test_data = {"test": "data"}

        success = cache.save_to_cache(test_data)

        assert success
        assert cache.cache_file.exists()

        saved_data = json.loads(cache.cache_file.read_text())
        assert saved_data == test_data

    async def test_get_pricing_data_cache_hit(self, tmp_path: Path) -> None:
        """Test getting pricing data from valid cache."""
        cache = PricingCache(cache_dir=str(tmp_path))
        test_data = {"test": "data"}

        # Save data to cache
        cache.save_to_cache(test_data)

        # Get data (should come from cache)
        retrieved_data = await cache.get_pricing_data()

        assert retrieved_data == test_data

    async def test_get_pricing_data_cache_miss_download_success(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test getting pricing data with cache miss but successful download."""
        test_data = {"test": "data"}

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))

        # Get data (should download and cache)
        retrieved_data = await cache.get_pricing_data()

        assert retrieved_data == test_data
        assert cache.cache_file.exists()

    def test_clear_cache_success(self, tmp_path: Path) -> None:
        """Test successful cache clearing."""
        cache = PricingCache(cache_dir=str(tmp_path))

        # Create cache file
        cache.save_to_cache({"test": "data"})
        assert cache.cache_file.exists()

        # Clear cache
        success = cache.clear_cache()

        assert success
        assert not cache.cache_file.exists()

    def test_get_cache_info(self, tmp_path: Path) -> None:
        """Test getting cache information."""
        cache = PricingCache(cache_dir=str(tmp_path))

        # Test with no cache file
        info = cache.get_cache_info()

        assert "cache_file" in info
        assert "cache_dir" in info
        assert "source_url" in info
        assert "ttl_hours" in info
        assert "exists" in info
        assert "valid" in info
        assert info["exists"] is False
        assert info["valid"] is False

        # Test with cache file
        cache.save_to_cache({"test": "data"})
        info = cache.get_cache_info()

        assert info["exists"] is True
        assert info["valid"] is True
        assert "age_hours" in info
        assert "size_bytes" in info


@pytest.mark.unit
class TestPricingLoader:
    """Test pricing data loading and format conversion."""

    def test_extract_claude_models_success(self) -> None:
        """Test successful extraction of Claude models from LiteLLM data."""
        litellm_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
            },
            "claude-3-opus-20240229": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075,
            },
        }

        claude_models = PricingLoader.extract_claude_models(litellm_data)

        assert len(claude_models) == 2
        assert "claude-3-5-sonnet-20241022" in claude_models
        assert "claude-3-opus-20240229" in claude_models
        assert "gpt-4" not in claude_models

    def test_extract_claude_models_no_claude_models(self) -> None:
        """Test extraction when no Claude models exist."""
        litellm_data = {
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
            },
        }

        claude_models = PricingLoader.extract_claude_models(litellm_data)

        assert len(claude_models) == 0

    def test_convert_to_internal_format_success(self) -> None:
        """Test successful conversion to internal format."""
        claude_models = {
            "claude-3-5-sonnet-20241022": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            },
        }

        internal_format = PricingLoader.convert_to_internal_format(claude_models)

        assert len(internal_format) == 1
        pricing = internal_format["claude-3-5-sonnet-20241022"]

        assert pricing["input"] == Decimal("3.00")
        assert pricing["output"] == Decimal("15.00")
        assert pricing["cache_write"] == Decimal("3.75")
        assert pricing["cache_read"] == Decimal("0.30")

    def test_convert_to_internal_format_missing_required_fields(self) -> None:
        """Test conversion with missing required fields."""
        claude_models = {
            "claude-incomplete": {
                "input_cost_per_token": 0.000003,
                # Missing output_cost_per_token
            },
        }

        internal_format = PricingLoader.convert_to_internal_format(claude_models)

        assert len(internal_format) == 0

    def test_convert_to_internal_format_model_mapping(self) -> None:
        """Test conversion with model name mapping."""
        claude_models = {
            "claude-3-5-sonnet-latest": {
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
        }

        internal_format = PricingLoader.convert_to_internal_format(claude_models)

        # Should be mapped to canonical name
        assert "claude-3-5-sonnet-20241022" in internal_format
        assert "claude-3-5-sonnet-latest" not in internal_format

    def test_load_pricing_from_data_success(self) -> None:
        """Test successful loading and conversion from LiteLLM data."""
        litellm_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            },
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.00003,
                "output_cost_per_token": 0.00006,
            },
        }

        pricing_data = PricingLoader.load_pricing_from_data(litellm_data)

        assert pricing_data is not None
        assert len(pricing_data) == 1
        assert "claude-3-5-sonnet-20241022" in pricing_data

    def test_validate_pricing_data_success(self) -> None:
        """Test successful validation of pricing data."""
        pricing_data = {
            "claude-3-5-sonnet-20241022": {
                "input": Decimal("3.00"),
                "output": Decimal("15.00"),
                "cache_read": Decimal("0.30"),
                "cache_write": Decimal("3.75"),
            },
        }

        is_valid = PricingLoader.validate_pricing_data(pricing_data)

        assert is_valid

    def test_validate_pricing_data_missing_required_fields(self) -> None:
        """Test validation with missing required fields."""
        pricing_data = {
            "claude-incomplete": {
                "input": Decimal("3.00"),
                # Missing "output"
            },
        }

        is_valid = PricingLoader.validate_pricing_data(pricing_data)

        assert not is_valid

    def test_get_canonical_model_name(self) -> None:
        """Test getting canonical model name."""
        # Test alias mapping
        canonical = PricingLoader.get_canonical_model_name("claude-3-5-sonnet-latest")
        assert canonical == "claude-3-5-sonnet-20241022"

        # Test already canonical name
        canonical = PricingLoader.get_canonical_model_name("claude-3-5-sonnet-20241022")
        assert canonical == "claude-3-5-sonnet-20241022"

        # Test unknown model
        canonical = PricingLoader.get_canonical_model_name("unknown-model")
        assert canonical == "unknown-model"


@pytest.mark.unit
class TestPricingUpdater:
    """Test pricing updater functionality."""

    def test_pricing_updater_initialization(self, tmp_path: Path) -> None:
        """Test pricing updater initialization."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(
            cache=cache, auto_update=False, fallback_to_embedded=False
        )

        assert updater.cache == cache
        assert updater.auto_update is False
        assert updater.fallback_to_embedded is False
        assert updater.memory_cache_ttl == 300

    async def test_get_current_pricing_memory_cache_hit(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test getting pricing from memory cache."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, memory_cache_ttl=60)

        # Load pricing once
        pricing1 = await updater.get_current_pricing()

        # Clear HTTP mock to ensure second call doesn't make network request
        httpx_mock.reset()

        # Should use memory cache on second call (no network request)
        pricing2 = await updater.get_current_pricing()

        assert pricing1 == pricing2
        assert pricing1 is not None
        assert len(pricing1) == 1
        assert "claude-3-5-sonnet-20241022" in pricing1

    async def test_get_current_pricing_fallback_to_embedded(
        self, tmp_path: Path
    ) -> None:
        """Test getting pricing with fallback to embedded when external fails."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)

        # No HTTP mock = network failure, should fall back to embedded
        pricing = await updater.get_current_pricing()

        # Should have embedded pricing data
        assert pricing is not None
        assert len(pricing) > 0
        assert "claude-3-5-sonnet-20241022" in pricing

    def test_get_embedded_pricing(self, tmp_path: Path) -> None:
        """Test getting embedded pricing data."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        embedded_pricing = updater._get_embedded_pricing()

        assert len(embedded_pricing) > 0
        assert "claude-3-5-sonnet-20241022" in embedded_pricing
        assert "claude-3-5-haiku-20241022" in embedded_pricing
        assert "claude-3-opus-20240229" in embedded_pricing

    async def test_force_refresh(self, tmp_path: Path, httpx_mock: HTTPXMock) -> None:
        """Test force refresh functionality."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        # Add two responses since force_refresh might call get_current_pricing after refresh
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)

        success = await updater.force_refresh()

        assert success
        assert cache.cache_file.exists()

    def test_clear_cache(self, tmp_path: Path) -> None:
        """Test clearing cache."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)

        # Create cache file
        cache.save_to_cache({"test": "data"})
        assert cache.cache_file.exists()

        success = updater.clear_cache()

        assert success
        assert not cache.cache_file.exists()

    async def test_get_pricing_info(self, tmp_path: Path) -> None:
        """Test getting pricing information."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)

        info = await updater.get_pricing_info()

        assert "models_loaded" in info
        assert "model_names" in info
        assert "auto_update" in info
        assert "fallback_to_embedded" in info
        assert "has_cached_pricing" in info
        assert info["auto_update"] is True
        assert info["fallback_to_embedded"] is True

    async def test_validate_external_source_success(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Test successful external source validation."""
        raw_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=raw_data,
            status_code=200,
        )

        updater = PricingUpdater()
        is_valid = await updater.validate_external_source()

        assert is_valid

    async def test_validate_external_source_failure(
        self, httpx_mock: HTTPXMock
    ) -> None:
        """Test external source validation failure."""
        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            status_code=404,
        )

        updater = PricingUpdater()
        is_valid = await updater.validate_external_source()

        assert not is_valid


@pytest.mark.unit
class TestCostCalculatorWithDynamicPricing:
    """Test cost calculator with dynamic pricing integration."""

    def test_cost_calculator_initialization_with_dynamic_pricing(
        self, tmp_path: Path
    ) -> None:
        """Test cost calculator initialization with dynamic pricing."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        calculator = CostCalculator(
            pricing_updater=updater, enable_dynamic_pricing=True
        )

        assert calculator._pricing_updater == updater
        assert calculator._enable_dynamic_pricing is True

    def test_cost_calculator_initialization_no_dynamic_pricing(self) -> None:
        """Test cost calculator initialization without dynamic pricing."""
        calculator = CostCalculator(enable_dynamic_pricing=False)

        assert calculator._enable_dynamic_pricing is False

    async def test_get_model_pricing_dynamic_pricing(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test getting model pricing from dynamic pricing."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        calculator = CostCalculator(pricing_updater=updater)

        pricing = await calculator.get_model_pricing("claude-3-5-sonnet-20241022")

        assert pricing.input == Decimal("3.00")
        assert pricing.output == Decimal("15.00")
        assert pricing.cache_read == Decimal("0.30")
        assert pricing.cache_write == Decimal("3.75")

    async def test_get_model_pricing_canonical_name_mapping(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test getting model pricing with canonical name mapping."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        calculator = CostCalculator(pricing_updater=updater)

        # Request with alias should map to canonical name
        pricing = await calculator.get_model_pricing("claude-3-5-sonnet-latest")

        assert pricing.input == Decimal("3.00")  # Should get mapped pricing

    async def test_get_model_pricing_custom_pricing_priority(
        self, tmp_path: Path
    ) -> None:
        """Test that custom pricing takes priority over dynamic pricing."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        # Add custom pricing
        custom_pricing = {
            "input": 5.0,
            "output": 20.0,
            "cache_read": 1.0,
            "cache_write": 5.0,
        }
        calculator.add_custom_pricing("claude-3-5-sonnet-20241022", custom_pricing)

        pricing = await calculator.get_model_pricing("claude-3-5-sonnet-20241022")

        # Should use custom pricing, not dynamic
        assert pricing.input == Decimal("5.0")
        assert pricing.output == Decimal("20.0")

    async def test_get_model_pricing_unknown_model_fallback(
        self, tmp_path: Path
    ) -> None:
        """Test getting pricing for unknown model falls back to default."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        pricing = await calculator.get_model_pricing("unknown-model")

        # Should use default pricing
        from ccproxy.pricing.models import ModelPricing

        expected_default = ModelPricing(**calculator.DEFAULT_PRICING)
        assert pricing == expected_default

    async def test_calculate_cost_with_dynamic_pricing(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test cost calculation with dynamic pricing."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            }
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=test_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        calculator = CostCalculator(pricing_updater=updater)

        cost_metric = await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=100,
            cache_write_tokens=200,
        )

        # Verify calculations with dynamic pricing
        expected_input_cost = 3.00 * 1000 / 1_000_000  # $0.003000
        expected_output_cost = 15.00 * 500 / 1_000_000  # $0.007500
        expected_cache_read_cost = 0.30 * 100 / 1_000_000  # $0.000030
        expected_cache_write_cost = 3.75 * 200 / 1_000_000  # $0.000750
        expected_total = (
            expected_input_cost
            + expected_output_cost
            + expected_cache_read_cost
            + expected_cache_write_cost
        )

        assert abs(cost_metric.input_cost - expected_input_cost) < 0.000001
        assert abs(cost_metric.output_cost - expected_output_cost) < 0.000001
        assert abs(cost_metric.cache_read_cost - expected_cache_read_cost) < 0.000001
        assert abs(cost_metric.cache_write_cost - expected_cache_write_cost) < 0.000001
        assert abs(cost_metric.total_cost - expected_total) < 0.000001

    async def test_calculate_cost_with_sdk_comparison(self, tmp_path: Path) -> None:
        """Test cost calculation with SDK cost comparison."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        sdk_total_cost = 0.010000  # Example SDK cost

        cost_metric = await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
            sdk_total_cost=sdk_total_cost,
        )

        assert cost_metric.sdk_total_cost == sdk_total_cost
        assert cost_metric.cost_difference is not None
        assert cost_metric.cost_accuracy_percentage is not None

    async def test_compare_model_costs_with_dynamic_pricing(
        self, tmp_path: Path
    ) -> None:
        """Test comparing costs across models with dynamic pricing."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        models = ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
        costs = await calculator.compare_model_costs(
            models=models,
            input_tokens=1000,
            output_tokens=500,
        )

        assert len(costs) == 2
        assert "claude-3-5-sonnet-20241022" in costs
        assert "claude-3-5-haiku-20241022" in costs

        # Sonnet should be more expensive than Haiku
        assert costs["claude-3-5-sonnet-20241022"] > costs["claude-3-5-haiku-20241022"]

    async def test_get_supported_models_with_dynamic_pricing(
        self, tmp_path: Path
    ) -> None:
        """Test getting supported models with dynamic pricing."""
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        supported_models = await calculator.get_supported_models()

        assert isinstance(supported_models, list)
        assert "claude-3-5-sonnet-20241022" in supported_models
        assert "claude-3-5-haiku-20241022" in supported_models


@pytest.mark.unit
class TestPricingIntegration:
    """Test integration between pricing components."""

    async def test_end_to_end_pricing_flow(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test complete end-to-end pricing flow."""
        # Mock LiteLLM data
        litellm_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            },
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=litellm_data,
            status_code=200,
        )

        # Create components
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache)
        calculator = CostCalculator(pricing_updater=updater)

        # Calculate cost (should trigger download and caching)
        cost_metric = await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
        )

        # Verify cost calculation worked
        assert cost_metric.total_cost > 0
        assert cost_metric.model == "claude-3-5-sonnet-20241022"

        # Verify cache file was created
        assert cache.cache_file.exists()

        # Verify second calculation uses cache (no HTTP call)
        httpx_mock.reset()

        cost_metric2 = await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should get same result
        assert cost_metric2.total_cost == cost_metric.total_cost

    async def test_pricing_error_handling_and_fallback(self, tmp_path: Path) -> None:
        """Test pricing error handling and fallback to embedded pricing."""
        # Create components with no network access (will fail)
        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, fallback_to_embedded=True)
        calculator = CostCalculator(pricing_updater=updater)

        # Calculate cost (should fall back to embedded pricing)
        cost_metric = await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should still work with embedded pricing
        assert cost_metric.total_cost > 0
        assert cost_metric.model == "claude-3-5-sonnet-20241022"

    async def test_pricing_memory_cache_performance(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """Test that memory caching improves performance."""
        litellm_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            },
        }

        httpx_mock.add_response(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
            json=litellm_data,
            status_code=200,
        )

        cache = PricingCache(cache_dir=str(tmp_path))
        updater = PricingUpdater(cache=cache, memory_cache_ttl=60)  # 1 minute cache
        calculator = CostCalculator(pricing_updater=updater)

        # First call - loads from network
        start_time = time.time()
        await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500
        )
        first_call_time = time.time() - start_time

        # Second call - uses memory cache
        start_time = time.time()
        await calculator.calculate_cost(
            model="claude-3-5-sonnet-20241022", input_tokens=2000, output_tokens=1000
        )
        second_call_time = time.time() - start_time

        # Second call should be significantly faster
        assert second_call_time < first_call_time * 0.5  # At least 50% faster
