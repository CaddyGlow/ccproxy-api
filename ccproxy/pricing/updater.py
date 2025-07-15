"""Pricing updater for managing periodic refresh of pricing data."""

import asyncio
from decimal import Decimal
from typing import Any

from structlog import get_logger

from .cache import PricingCache
from .loader import PricingLoader
from .models import PricingData


logger = get_logger(__name__)


class PricingUpdater:
    """Manages periodic updates of pricing data."""

    def __init__(
        self,
        cache: PricingCache | None = None,
        auto_update: bool = True,
        fallback_to_embedded: bool = True,
        memory_cache_ttl: int = 300,  # 5 minutes in memory cache
    ) -> None:
        """Initialize pricing updater.

        Args:
            cache: Pricing cache instance (creates default if None)
            auto_update: Whether to auto-update stale cache
            fallback_to_embedded: Whether to fallback to embedded pricing on failure
            memory_cache_ttl: Time to live for in-memory cache in seconds
        """
        self.cache = cache or PricingCache()
        self.auto_update = auto_update
        self.fallback_to_embedded = fallback_to_embedded
        self.memory_cache_ttl = memory_cache_ttl
        self._cached_pricing: PricingData | None = None
        self._last_load_time: float = 0
        self._last_file_check_time: float = 0
        self._cached_file_mtime: float = 0

    async def get_current_pricing(
        self, force_refresh: bool = False
    ) -> PricingData | None:
        """Get current pricing data with automatic updates.

        Args:
            force_refresh: Force refresh even if cache is valid

        Returns:
            Current pricing data as PricingData model
        """
        import time

        current_time = time.time()

        # Return cached pricing if recent and not forced
        if (
            not force_refresh
            and self._cached_pricing is not None
            and (current_time - self._last_load_time) < self.memory_cache_ttl
        ):
            # Only check file changes every 30 seconds to reduce I/O
            if (current_time - self._last_file_check_time) > 30:
                if self._has_cache_file_changed():
                    logger.info("Cache file has changed, reloading pricing data")
                    # File changed, need to reload
                    pricing_data = await self._load_pricing_data()
                    self._cached_pricing = pricing_data
                    self._last_load_time = current_time
                    return pricing_data
                self._last_file_check_time = current_time

            return self._cached_pricing

        # Check if we need to refresh
        should_refresh = force_refresh or (
            self.auto_update and not self.cache.is_cache_valid()
        )

        if should_refresh:
            logger.info("Refreshing pricing data from external source")
            await self._refresh_pricing()

        # Load pricing data
        pricing_data = await self._load_pricing_data()

        # Cache the result
        self._cached_pricing = pricing_data
        self._last_load_time = current_time
        self._last_file_check_time = current_time

        return pricing_data

    def _has_cache_file_changed(self) -> bool:
        """Check if the cache file has changed since last load.

        Returns:
            True if file has changed or doesn't exist
        """
        try:
            if not self.cache.cache_file.exists():
                return self._cached_file_mtime != 0  # File was deleted

            current_mtime = self.cache.cache_file.stat().st_mtime
            if current_mtime != self._cached_file_mtime:
                self._cached_file_mtime = current_mtime
                return True
            return False
        except OSError:
            # If we can't check, assume it changed
            return True

    async def _refresh_pricing(self) -> bool:
        """Refresh pricing data from external source.

        Returns:
            True if refresh was successful
        """
        try:
            logger.info("Refreshing pricing data from external source")

            # Download fresh data
            raw_data = await self.cache.download_pricing_data()
            if raw_data is None:
                logger.error("Failed to download fresh pricing data")
                return False

            # Save to cache
            if not self.cache.save_to_cache(raw_data):
                logger.error("Failed to save pricing data to cache")
                return False

            logger.info("Successfully refreshed pricing data")
            return True

        except Exception as e:
            logger.error(f"Failed to refresh pricing data: {e}")
            return False

    async def _load_pricing_data(self) -> PricingData | None:
        """Load pricing data from available sources.

        Returns:
            Pricing data as PricingData model
        """
        # Try to get data from cache or download
        raw_data = await self.cache.get_pricing_data()

        if raw_data is not None:
            # Load and validate pricing data using Pydantic
            pricing_data = PricingLoader.load_pricing_from_data(raw_data, verbose=False)

            if pricing_data:
                logger.info(
                    f"Loaded pricing for {len(pricing_data)} Claude models from external source"
                )
                return pricing_data
            else:
                logger.warning("External pricing data failed validation")

        # Fallback to embedded pricing
        if self.fallback_to_embedded:
            logger.info("Using embedded pricing data as fallback")
            return self._get_embedded_pricing()
        else:
            logger.error("No valid pricing data available and fallback disabled")
            return None

    def _get_embedded_pricing(self) -> PricingData:
        """Get embedded (hardcoded) pricing as fallback.

        Returns:
            Embedded pricing data as PricingData model
        """
        # This is the current hardcoded pricing from CostCalculator
        embedded_data = {
            "claude-3-5-sonnet-20241022": {
                "input": Decimal("3.00"),
                "output": Decimal("15.00"),
                "cache_read": Decimal("0.30"),
                "cache_write": Decimal("3.75"),
            },
            "claude-3-5-haiku-20241022": {
                "input": Decimal("0.25"),
                "output": Decimal("1.25"),
                "cache_read": Decimal("0.03"),
                "cache_write": Decimal("0.30"),
            },
            "claude-3-opus-20240229": {
                "input": Decimal("15.00"),
                "output": Decimal("75.00"),
                "cache_read": Decimal("1.50"),
                "cache_write": Decimal("18.75"),
            },
            "claude-3-sonnet-20240229": {
                "input": Decimal("3.00"),
                "output": Decimal("15.00"),
                "cache_read": Decimal("0.30"),
                "cache_write": Decimal("3.75"),
            },
            "claude-3-haiku-20240307": {
                "input": Decimal("0.25"),
                "output": Decimal("1.25"),
                "cache_read": Decimal("0.03"),
                "cache_write": Decimal("0.30"),
            },
        }

        # Create PricingData from embedded data
        return PricingData.from_dict(embedded_data)

    async def force_refresh(self) -> bool:
        """Force a refresh of pricing data.

        Returns:
            True if refresh was successful
        """
        logger.info("Force refreshing pricing data")

        # Clear cached pricing
        self._cached_pricing = None
        self._last_load_time = 0

        # Refresh from external source
        success = await self._refresh_pricing()

        if success:
            # Reload pricing data
            await self.get_current_pricing(force_refresh=True)

        return success

    def clear_cache(self) -> bool:
        """Clear all cached pricing data.

        Returns:
            True if cache was cleared successfully
        """
        logger.info("Clearing pricing cache")

        # Clear in-memory cache
        self._cached_pricing = None
        self._last_load_time = 0

        # Clear file cache
        return self.cache.clear_cache()

    async def get_pricing_info(self) -> dict[str, Any]:
        """Get information about current pricing state.

        Returns:
            Dictionary with pricing information
        """
        cache_info = self.cache.get_cache_info()

        pricing_data = await self.get_current_pricing()

        return {
            "models_loaded": len(pricing_data) if pricing_data else 0,
            "model_names": pricing_data.model_names() if pricing_data else [],
            "auto_update": self.auto_update,
            "fallback_to_embedded": self.fallback_to_embedded,
            "has_cached_pricing": self._cached_pricing is not None,
        }

    async def validate_external_source(self) -> bool:
        """Validate that external pricing source is accessible.

        Returns:
            True if external source is accessible and has valid data
        """
        try:
            logger.debug("Validating external pricing source")

            # Try to download data
            raw_data = await self.cache.download_pricing_data(timeout=10)
            if raw_data is None:
                return False

            # Try to parse Claude models
            claude_models = PricingLoader.extract_claude_models(raw_data)
            if not claude_models:
                logger.warning("No Claude models found in external source")
                return False

            # Try to load and validate using Pydantic
            pricing_data = PricingLoader.load_pricing_from_data(raw_data, verbose=False)
            if not pricing_data:
                logger.warning("Failed to load and validate external pricing data")
                return False

            logger.info(
                f"External source is valid with {len(pricing_data)} Claude models"
            )
            return True

        except Exception as e:
            logger.error(f"External source validation failed: {e}")
            return False
