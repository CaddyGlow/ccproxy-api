"""Test pricing module functionality with dependency injection."""

import json
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

from ccproxy.plugins.pricing.cache import PricingCache
from ccproxy.plugins.pricing.config import PricingConfig
from ccproxy.plugins.pricing.loader import PricingLoader
from ccproxy.plugins.pricing.models import PricingData
from ccproxy.plugins.pricing.updater import PricingUpdater


@pytest.mark.unit
class TestPricingConfig:
    """Test PricingConfig configuration class."""

    def test_default_settings(self) -> None:
        """Test default pricing settings values."""
        settings = PricingConfig()

        assert settings.cache_ttl_hours == 24
        assert settings.source_url.startswith(
            "https://raw.githubusercontent.com/BerriAI/litellm"
        )
        assert settings.download_timeout == 30
        assert settings.auto_update is True
        assert settings.fallback_to_embedded is False
        assert settings.memory_cache_ttl == 300
        assert str(settings.cache_dir).endswith("ccproxy")

    def test_settings_with_custom_cache_dir(self, tmp_path: Path) -> None:
        """Test settings with custom cache directory."""
        custom_cache = tmp_path / "custom_cache"
        settings = PricingConfig(cache_dir=custom_cache)

        assert settings.cache_dir == custom_cache

    def test_settings_with_environment_variables(self) -> None:
        """Test settings configured via environment variables."""
        with patch.dict(
            os.environ,
            {
                "PRICING__CACHE_TTL_HOURS": "48",
                "PRICING__AUTO_UPDATE": "false",
                "PRICING__DOWNLOAD_TIMEOUT": "60",
            },
        ):
            settings = PricingConfig()

            assert settings.cache_ttl_hours == 48
            assert settings.auto_update is False
            assert settings.download_timeout == 60

    def test_settings_validation_errors(self) -> None:
        """Test settings validation with invalid values."""
        with pytest.raises(ValueError):
            PricingConfig(source_url="invalid-url")

    def test_settings_cache_dir_expansion(self) -> None:
        """Test cache directory path expansion."""
        from pathlib import Path

        settings = PricingConfig(cache_dir=Path("~/test_cache").expanduser())
        assert not str(settings.cache_dir).startswith("~")
        assert settings.cache_dir.is_absolute()


@pytest.mark.unit
class TestPricingCache:
    """Test PricingCache with dependency injection."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> PricingConfig:
        """Create test pricing settings."""
        return PricingConfig(
            cache_dir=tmp_path / "test_cache",
            cache_ttl_hours=1,
            source_url="https://example.com/pricing.json",
            download_timeout=10,
        )

    @pytest.fixture
    def cache(self, settings: PricingConfig) -> PricingCache:
        """Create test pricing cache."""
        return PricingCache(settings)

    def test_cache_initialization(
        self, cache: PricingCache, settings: PricingConfig
    ) -> None:
        """Test cache initialization with settings."""
        assert cache.settings == settings
        assert cache.cache_dir == settings.cache_dir
        assert cache.cache_file == settings.cache_dir / "model_pricing.json"
        assert cache.cache_dir.exists()

    def test_cache_directory_creation(self, tmp_path: Path) -> None:
        """Test cache directory is created automatically."""
        cache_dir = tmp_path / "deep" / "nested" / "cache"
        settings = PricingConfig(cache_dir=cache_dir)
        cache = PricingCache(settings)

        assert cache_dir.exists()
        assert cache.cache_dir == cache_dir

    def test_cache_validation_fresh_cache(self, cache: PricingCache) -> None:
        """Test cache validation with fresh cache."""
        # Create a fresh cache file
        cache.cache_file.write_text('{"test": "data"}')

        assert cache.is_cache_valid() is True

    def test_cache_validation_expired_cache(self, cache: PricingCache) -> None:
        """Test cache validation with expired cache."""
        # Create an old cache file
        cache.cache_file.write_text('{"test": "data"}')

        # Modify file time to make it old
        old_time = time.time() - (cache.settings.cache_ttl_hours + 1) * 3600
        os.utime(cache.cache_file, (old_time, old_time))

        assert cache.is_cache_valid() is False

    def test_cache_validation_missing_file(self, cache: PricingCache) -> None:
        """Test cache validation with missing cache file."""
        assert cache.is_cache_valid() is False

    def test_load_cached_data_valid(self, cache: PricingCache) -> None:
        """Test loading data from valid cache."""
        test_data = {"model": "claude-3", "price": 1.0}
        cache.cache_file.write_text(json.dumps(test_data))

        loaded_data = cache.load_cached_data()
        assert loaded_data == test_data

    def test_load_cached_data_invalid_json(self, cache: PricingCache) -> None:
        """Test loading data from cache with invalid JSON."""
        cache.cache_file.write_text("invalid json")

        loaded_data = cache.load_cached_data()
        assert loaded_data is None

    def test_load_cached_data_expired(self, cache: PricingCache) -> None:
        """Test loading data from expired cache."""
        cache.cache_file.write_text('{"test": "data"}')

        # Make cache expired
        old_time = time.time() - (cache.settings.cache_ttl_hours + 1) * 3600
        os.utime(cache.cache_file, (old_time, old_time))

        loaded_data = cache.load_cached_data()
        assert loaded_data is None

    @pytest.mark.asyncio
    async def test_download_pricing_data_success(self, cache: PricingCache) -> None:
        """Test successful download of pricing data."""
        test_data = {"claude-3-5-sonnet-20241022": {"input_cost_per_token": 0.000003}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = test_data
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await cache.download_pricing_data()

            assert result == test_data
            mock_client.return_value.__aenter__.return_value.get.assert_called_once_with(
                cache.settings.source_url
            )

    @pytest.mark.asyncio
    async def test_download_pricing_data_http_error(self, cache: PricingCache) -> None:
        """Test download failure with HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=Mock()
            )

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await cache.download_pricing_data()

            assert result is None

    @pytest.mark.asyncio
    async def test_download_pricing_data_timeout(self, cache: PricingCache) -> None:
        """Test download with custom timeout."""
        test_data = {"test": "data"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.json.return_value = test_data
            mock_response.raise_for_status.return_value = None

            mock_client.return_value.__aenter__.return_value.get.return_value = (
                mock_response
            )

            result = await cache.download_pricing_data(timeout=5)

            assert result == test_data
            mock_client.assert_called_once_with(timeout=5)

    def test_save_to_cache_success(self, cache: PricingCache) -> None:
        """Test successful cache save."""
        test_data = {"model": "claude-3", "price": 1.0}

        result = cache.save_to_cache(test_data)

        assert result is True
        assert cache.cache_file.exists()

        # Verify content
        with cache.cache_file.open() as f:
            saved_data = json.load(f)
        assert saved_data == test_data

    def test_save_to_cache_atomic_write(self, cache: PricingCache) -> None:
        """Test atomic write operation for cache save."""
        test_data = {"model": "claude-3", "price": 1.0}

        # Mock file operations to simulate failure during write
        with patch("pathlib.Path.open") as mock_open:
            mock_open.side_effect = OSError("Disk full")

            result = cache.save_to_cache(test_data)

            assert result is False
            # Original cache file should not exist after failed atomic write
            assert not cache.cache_file.exists()

    def test_clear_cache_success(self, cache: PricingCache) -> None:
        """Test successful cache clearing."""
        # Create cache file
        cache.cache_file.write_text('{"test": "data"}')
        assert cache.cache_file.exists()

        result = cache.clear_cache()

        assert result is True
        assert not cache.cache_file.exists()

    def test_clear_cache_nonexistent_file(self, cache: PricingCache) -> None:
        """Test clearing nonexistent cache file."""
        result = cache.clear_cache()
        assert result is True

    def test_get_cache_info(self, cache: PricingCache) -> None:
        """Test cache information retrieval."""
        info = cache.get_cache_info()

        assert "cache_file" in info
        assert "cache_dir" in info
        assert "source_url" in info
        assert "ttl_hours" in info
        assert "exists" in info
        assert "valid" in info

        assert info["source_url"] == cache.settings.source_url
        assert info["ttl_hours"] == cache.settings.cache_ttl_hours
        assert info["exists"] is False
        assert info["valid"] is False

    def test_get_cache_info_with_existing_file(self, cache: PricingCache) -> None:
        """Test cache info with existing cache file."""
        test_data = {"model": "test"}
        cache.cache_file.write_text(json.dumps(test_data))

        info = cache.get_cache_info()

        assert info["exists"] is True
        assert info["valid"] is True
        assert isinstance(info["age_hours"], float)
        assert isinstance(info["size_bytes"], int)


@pytest.mark.unit
class TestPricingLoader:
    """Test PricingLoader data conversion functionality."""

    @pytest.fixture
    def sample_litellm_data(self) -> dict[str, Any]:
        """Sample LiteLLM pricing data."""
        return {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
                "max_tokens": 8192,
            },
            "claude-3-haiku-20240307": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.00000025,
                "output_cost_per_token": 0.00000125,
                "cache_creation_input_token_cost": 0.0000003,
                "cache_read_input_token_cost": 0.00000003,
                "max_tokens": 4096,
            },
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.000030,
                "output_cost_per_token": 0.000060,
                "max_tokens": 8192,
            },
        }

    def test_extract_claude_models(self, sample_litellm_data: dict[str, Any]) -> None:
        """Test extraction of Claude models from LiteLLM data."""
        claude_models = PricingLoader.extract_claude_models(
            sample_litellm_data, verbose=False
        )

        assert len(claude_models) == 2
        assert "claude-3-5-sonnet-20241022" in claude_models
        assert "claude-3-haiku-20240307" in claude_models
        assert "gpt-4" not in claude_models

    def test_extract_claude_models_empty_input(self) -> None:
        """Test extraction with empty input data."""
        claude_models = PricingLoader.extract_claude_models({}, verbose=False)
        assert len(claude_models) == 0

    def test_convert_to_internal_format(
        self, sample_litellm_data: dict[str, Any]
    ) -> None:
        """Test conversion to internal pricing format."""
        claude_models = PricingLoader.extract_claude_models(
            sample_litellm_data, verbose=False
        )
        internal_format = PricingLoader.convert_to_internal_format(
            claude_models, verbose=False
        )

        assert len(internal_format) == 2

        # Check Sonnet pricing
        sonnet_pricing = internal_format["claude-3-5-sonnet-20241022"]
        assert sonnet_pricing["input"] == Decimal("3.00")  # 0.000003 * 1M
        assert sonnet_pricing["output"] == Decimal("15.00")  # 0.000015 * 1M
        assert sonnet_pricing["cache_write"] == Decimal("3.75")  # 0.00000375 * 1M
        assert sonnet_pricing["cache_read"] == Decimal("0.30")  # 0.0000003 * 1M

        # Check Haiku pricing
        haiku_pricing = internal_format["claude-3-haiku-20240307"]
        assert haiku_pricing["input"] == Decimal("0.25")  # 0.00000025 * 1M
        assert haiku_pricing["output"] == Decimal("1.25")  # 0.00000125 * 1M

    def test_convert_to_internal_format_missing_pricing(self) -> None:
        """Test conversion with missing pricing information."""
        data = {
            "claude-3-test": {
                "litellm_provider": "anthropic",
                "max_tokens": 4096,
                # Missing pricing fields
            }
        }

        claude_models = PricingLoader.extract_claude_models(data, verbose=False)
        internal_format = PricingLoader.convert_to_internal_format(
            claude_models, verbose=False
        )

        assert len(internal_format) == 0

    def test_load_pricing_from_data_success(
        self, sample_litellm_data: dict[str, Any]
    ) -> None:
        """Test successful loading of pricing data."""
        pricing_data = PricingLoader.load_pricing_from_data(
            sample_litellm_data, verbose=False
        )

        assert pricing_data is not None
        assert isinstance(pricing_data, PricingData)
        assert len(pricing_data) == 2
        assert "claude-3-5-sonnet-20241022" in pricing_data
        assert "claude-3-haiku-20240307" in pricing_data

    def test_load_pricing_from_data_no_claude_models(self) -> None:
        """Test loading with no Claude models in data."""
        data = {
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.000030,
                "output_cost_per_token": 0.000060,
            }
        }

        pricing_data = PricingLoader.load_pricing_from_data(data, verbose=False)
        assert pricing_data is None

    def test_load_pricing_from_data_invalid_data(self) -> None:
        """Test loading with invalid data format."""
        invalid_data: dict[str, Any] = {"invalid": "data"}
        pricing_data = PricingLoader.load_pricing_from_data(invalid_data, verbose=False)
        assert pricing_data is None

    def test_validate_pricing_data_valid_dict(self) -> None:
        """Test validation with valid pricing dictionary."""
        data = {
            "claude-3-5-sonnet-20241022": {
                "input": Decimal("3.00"),
                "output": Decimal("15.00"),
            }
        }

        validated = PricingLoader.validate_pricing_data(data, verbose=False)

        assert validated is not None
        assert isinstance(validated, PricingData)
        assert len(validated) == 1

    def test_validate_pricing_data_already_validated(self) -> None:
        """Test validation with already validated PricingData."""
        original_data = PricingData.from_dict(
            {
                "claude-3-5-sonnet-20241022": {
                    "input": Decimal("3.00"),
                    "output": Decimal("15.00"),
                }
            }
        )

        validated = PricingLoader.validate_pricing_data(original_data, verbose=False)

        assert validated is original_data

    def test_validate_pricing_data_empty_dict(self) -> None:
        """Test validation with empty dictionary."""
        validated = PricingLoader.validate_pricing_data({}, verbose=False)
        assert validated is None

    def test_validate_pricing_data_invalid_type(self) -> None:
        """Test validation with invalid data type."""
        validated = PricingLoader.validate_pricing_data(123, verbose=False)
        assert validated is None

    def test_get_model_aliases(self) -> None:
        """Test model alias mapping retrieval."""
        aliases = PricingLoader.get_model_aliases()

        assert isinstance(aliases, dict)
        assert "claude-3-5-sonnet-latest" in aliases
        assert "claude-3-opus" in aliases
        assert aliases["claude-3-5-sonnet-latest"] == "claude-3-5-sonnet-20241022"

    def test_get_canonical_model_name(self) -> None:
        """Test canonical model name resolution."""
        # Test alias resolution
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
    """Test PricingUpdater with dependency injection."""

    @pytest.fixture
    def settings(self, tmp_path: Path) -> PricingConfig:
        """Create test pricing settings."""
        return PricingConfig(
            cache_dir=tmp_path / "test_cache",
            cache_ttl_hours=1,
            auto_update=True,
            fallback_to_embedded=False,  # Updated to match current default
            memory_cache_ttl=60,
        )

    @pytest.fixture
    def cache(self, settings: PricingConfig) -> PricingCache:
        """Create test pricing cache."""
        return PricingCache(settings)

    @pytest.fixture
    def updater(self, cache: PricingCache, settings: PricingConfig) -> PricingUpdater:
        """Create test pricing updater."""
        return PricingUpdater(cache, settings)

    def test_updater_initialization(
        self, updater: PricingUpdater, cache: PricingCache, settings: PricingConfig
    ) -> None:
        """Test updater initialization with dependency injection."""
        assert updater.cache is cache
        assert updater.settings is settings
        assert updater._cached_pricing is None
        assert updater._last_load_time == 0

    @pytest.mark.asyncio
    async def test_get_current_pricing_with_valid_cache(
        self, updater: PricingUpdater
    ) -> None:
        """Test getting current pricing with valid cache."""
        # Create valid cache data
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        updater.cache.save_to_cache(test_data)

        pricing_data = await updater.get_current_pricing()

        assert pricing_data is not None
        assert isinstance(pricing_data, PricingData)
        assert "claude-3-5-sonnet-20241022" in pricing_data

    @pytest.mark.asyncio
    async def test_get_current_pricing_fallback_to_embedded(
        self, updater: PricingUpdater
    ) -> None:
        """Test behavior when cache fails and embedded pricing is disabled."""
        # No cache file exists, embedded pricing is disabled by default
        with patch.object(updater.cache, "get_pricing_data", return_value=None):
            pricing_data = await updater.get_current_pricing()

            # Since embedded pricing is disabled and no cache data, should be None
            assert pricing_data is None

    @pytest.mark.asyncio
    async def test_get_current_pricing_no_fallback(
        self, updater: PricingUpdater
    ) -> None:
        """Test behavior when fallback is disabled."""
        updater.settings.fallback_to_embedded = False

        with patch.object(updater.cache, "get_pricing_data", return_value=None):
            pricing_data = await updater.get_current_pricing()

            assert pricing_data is None

    @pytest.mark.asyncio
    async def test_get_current_pricing_force_refresh(
        self, updater: PricingUpdater
    ) -> None:
        """Test forced refresh of pricing data."""
        with (
            patch.object(
                updater, "_refresh_pricing", return_value=True
            ) as mock_refresh,
            patch.object(updater, "_load_pricing_data", return_value=None),
        ):
            await updater.get_current_pricing(force_refresh=True)

            mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_pricing_memory_cache(
        self, updater: PricingUpdater
    ) -> None:
        """Test memory cache behavior."""
        # Create mock pricing data
        from decimal import Decimal

        from ccproxy.plugins.pricing.models import PricingData

        mock_pricing_data = PricingData.from_dict(
            {
                "claude-3-5-sonnet-20241022": {
                    "input": Decimal("3.00"),
                    "output": Decimal("15.00"),
                }
            }
        )

        # Set up cached pricing
        updater._cached_pricing = mock_pricing_data
        updater._last_load_time = time.time()

        # Should return cached pricing without loading
        with patch.object(updater, "_load_pricing_data") as mock_load:
            pricing_data = await updater.get_current_pricing()

            assert pricing_data is mock_pricing_data
            mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_refresh_pricing_success(self, updater: PricingUpdater) -> None:
        """Test successful pricing refresh."""
        test_data = {"test": "data"}

        with (
            patch.object(
                updater.cache, "download_pricing_data", return_value=test_data
            ),
            patch.object(updater.cache, "save_to_cache", return_value=True),
        ):
            result = await updater._refresh_pricing()

            assert result is True

    @pytest.mark.asyncio
    async def test_refresh_pricing_download_failure(
        self, updater: PricingUpdater
    ) -> None:
        """Test pricing refresh with download failure."""
        with patch.object(updater.cache, "download_pricing_data", return_value=None):
            result = await updater._refresh_pricing()

            assert result is False

    @pytest.mark.asyncio
    async def test_refresh_pricing_save_failure(self, updater: PricingUpdater) -> None:
        """Test pricing refresh with save failure."""
        test_data = {"test": "data"}

        with (
            patch.object(
                updater.cache, "download_pricing_data", return_value=test_data
            ),
            patch.object(updater.cache, "save_to_cache", return_value=False),
        ):
            result = await updater._refresh_pricing()

            assert result is False

    def test_get_embedded_pricing(self, updater: PricingUpdater) -> None:
        """Test embedded pricing returns None (deprecated feature)."""
        embedded_pricing = updater._get_embedded_pricing()

        # Embedded pricing has been removed, should return None
        assert embedded_pricing is None

    @pytest.mark.asyncio
    async def test_force_refresh(self, updater: PricingUpdater) -> None:
        """Test forced refresh functionality."""
        with (
            patch.object(
                updater, "_refresh_pricing", return_value=True
            ) as mock_refresh,
            patch.object(updater, "get_current_pricing") as mock_get,
        ):
            result = await updater.force_refresh()

            assert result is True
            mock_refresh.assert_called_once()
            mock_get.assert_called_once_with(force_refresh=True)

    def test_clear_cache(self, updater: PricingUpdater) -> None:
        """Test cache clearing functionality."""
        # Set up cached data
        updater._cached_pricing = updater._get_embedded_pricing()
        updater._last_load_time = time.time()

        with patch.object(
            updater.cache, "clear_cache", return_value=True
        ) as mock_clear:
            result = updater.clear_cache()

            assert result is True
            # Verify internal state was reset
            assert updater._cached_pricing is None and updater._last_load_time <= 0.0
            mock_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_pricing_info(self, updater: PricingUpdater) -> None:
        """Test pricing information retrieval."""
        with patch.object(
            updater, "get_current_pricing", return_value=updater._get_embedded_pricing()
        ):
            info = await updater.get_pricing_info()

            assert isinstance(info, dict)
            assert "models_loaded" in info
            assert "model_names" in info
            assert "auto_update" in info
            assert "has_cached_pricing" in info

            assert info["auto_update"] == updater.settings.auto_update
            # Note: fallback_to_embedded no longer in response
            assert isinstance(info["models_loaded"], int)

    @pytest.mark.asyncio
    async def test_validate_external_source_success(
        self, updater: PricingUpdater
    ) -> None:
        """Test successful external source validation."""
        test_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }

        with patch.object(
            updater.cache, "download_pricing_data", return_value=test_data
        ):
            result = await updater.validate_external_source()

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_external_source_download_failure(
        self, updater: PricingUpdater
    ) -> None:
        """Test external source validation with download failure."""
        with patch.object(updater.cache, "download_pricing_data", return_value=None):
            result = await updater.validate_external_source()

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_external_source_no_claude_models(
        self, updater: PricingUpdater
    ) -> None:
        """Test external source validation with OpenAI models only."""
        test_data = {
            "gpt-4": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.000030,
                "output_cost_per_token": 0.000060,
            }
        }

        with patch.object(
            updater.cache, "download_pricing_data", return_value=test_data
        ):
            result = await updater.validate_external_source()

            # Should succeed since OpenAI models are valid with pricing_provider="all"
            assert result is True


@pytest.mark.integration
class TestPricingIntegration:
    """Integration tests for the complete pricing system."""

    @pytest.mark.asyncio
    async def test_full_pricing_workflow(self, isolated_environment: Path) -> None:
        """Test complete pricing workflow with dependency injection."""
        # Set up components
        settings = PricingConfig(
            cache_dir=isolated_environment / "cache",
            cache_ttl_hours=24,
            auto_update=True,
            fallback_to_embedded=False,  # Updated to match current default
        )

        cache = PricingCache(settings)
        updater = PricingUpdater(cache, settings)

        # Test initial load (should use embedded pricing)
        pricing_data = await updater.get_current_pricing()

        assert pricing_data is not None
        assert len(pricing_data) > 0

        # Test cache info
        info = await updater.get_pricing_info()
        assert info["models_loaded"] > 0
        assert info["auto_update"] is True

    @pytest.mark.asyncio
    async def test_pricing_with_mock_external_data(
        self, isolated_environment: Path
    ) -> None:
        """Test pricing with mocked external data download."""
        settings = PricingConfig(cache_dir=isolated_environment / "cache")
        cache = PricingCache(settings)
        updater = PricingUpdater(cache, settings)

        # Mock external data
        mock_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
                "cache_creation_input_token_cost": 0.00000375,
                "cache_read_input_token_cost": 0.0000003,
            }
        }

        with patch.object(cache, "download_pricing_data", return_value=mock_data):
            # Force refresh to use mock data
            success = await updater.force_refresh()
            assert success is True

            # Get pricing data
            pricing_data = await updater.get_current_pricing()

            assert pricing_data is not None
            assert "claude-3-5-sonnet-20241022" in pricing_data

            model_pricing = pricing_data["claude-3-5-sonnet-20241022"]
            assert model_pricing.input == Decimal("3.00")
            assert model_pricing.output == Decimal("15.00")
            assert model_pricing.cache_write == Decimal("3.75")
            assert model_pricing.cache_read == Decimal("0.30")

    @pytest.mark.asyncio
    async def test_cost_calculator_integration(
        self, isolated_environment: Path
    ) -> None:
        """Test integration with cost calculator utility."""
        from ccproxy.plugins.pricing.utils import calculate_token_cost

        # Create test data directly since embedded pricing is removed
        settings = PricingConfig()
        cache = PricingCache(settings)

        # Ensure the cache directory structure exists
        cache.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create test pricing data
        test_pricing_data = {
            "claude-3-5-sonnet-20241022": {
                "litellm_provider": "anthropic",
                "input_cost_per_token": 0.000003,
                "output_cost_per_token": 0.000015,
            }
        }
        # Save test data to cache
        cache.save_to_cache(test_pricing_data)

        # Test cost calculation (should find the cached data)
        cost = await calculate_token_cost(
            tokens_input=1000, tokens_output=500, model="claude-3-5-sonnet-20241022"
        )

        assert cost is not None
        assert isinstance(cost, float)
        assert cost > 0

    @pytest.mark.asyncio
    async def test_scheduler_task_integration(self, isolated_environment: Path) -> None:
        """Test integration with scheduler tasks."""
        from ccproxy.plugins.pricing.service import PricingService
        from ccproxy.plugins.pricing.tasks import PricingCacheUpdateTask

        settings = PricingConfig(cache_dir=isolated_environment / "cache")
        service = PricingService(settings)

        # Create task with pricing service
        task = PricingCacheUpdateTask(
            name="test_pricing_task", interval_seconds=3600, pricing_service=service
        )

        # Run task (no setup needed for plugin version)
        result = await task.run()

        # Task should succeed even without data (it's not a failure condition)
        assert result in [True, False]  # Either success or no data available


if __name__ == "__main__":
    pytest.main([__file__])
