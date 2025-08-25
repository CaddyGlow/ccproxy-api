"""Pricing service providing unified interface for pricing functionality."""

from decimal import Decimal
from typing import Any

import structlog

from .cache import PricingCache
from .config import PricingConfig
from .loader import PricingLoader
from .models import ModelPricing, PricingData
from .updater import PricingUpdater


logger = structlog.get_logger(__name__)


class PricingService:
    """Main service interface for pricing functionality."""

    def __init__(self, config: PricingConfig):
        """Initialize pricing service with configuration."""
        self.config = config
        self.cache = PricingCache(config)
        self.loader = PricingLoader()
        self.updater = PricingUpdater(self.cache, config)
        self._current_pricing: PricingData | None = None

    async def initialize(self) -> None:
        """Initialize the pricing service."""
        if not self.config.enabled:
            logger.info("pricing_service_disabled")
            return

        logger.info("pricing_service_initializing")

        # Force refresh on startup if configured
        if self.config.force_refresh_on_startup:
            await self.force_refresh_pricing()
        else:
            # Load current pricing data
            await self.get_current_pricing()

    async def get_current_pricing(
        self, force_refresh: bool = False
    ) -> PricingData | None:
        """Get current pricing data."""
        if not self.config.enabled:
            return None

        if force_refresh or self._current_pricing is None:
            self._current_pricing = await self.updater.get_current_pricing(
                force_refresh
            )

        return self._current_pricing

    async def get_model_pricing(self, model_name: str) -> ModelPricing | None:
        """Get pricing for specific model."""
        pricing_data = await self.get_current_pricing()
        if pricing_data is None:
            return None

        return pricing_data.get(model_name)

    async def calculate_cost(
        self,
        model_name: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> Decimal | None:
        """Calculate cost for token usage."""
        model_pricing = await self.get_model_pricing(model_name)
        if model_pricing is None:
            return None

        # Calculate cost per million tokens, then scale to actual tokens
        total_cost = Decimal("0")

        if input_tokens > 0:
            total_cost += (model_pricing.input * input_tokens) / Decimal("1000000")

        if output_tokens > 0:
            total_cost += (model_pricing.output * output_tokens) / Decimal("1000000")

        if cache_read_tokens > 0:
            total_cost += (model_pricing.cache_read * cache_read_tokens) / Decimal(
                "1000000"
            )

        if cache_write_tokens > 0:
            total_cost += (model_pricing.cache_write * cache_write_tokens) / Decimal(
                "1000000"
            )

        return total_cost

    async def force_refresh_pricing(self) -> bool:
        """Force refresh of pricing data."""
        if not self.config.enabled:
            return False

        success = await self.updater.force_refresh()
        if success:
            # Reload the current pricing data after successful refresh
            self._current_pricing = await self.updater.get_current_pricing(
                force_refresh=True
            )
            return True
        return False

    async def get_available_models(self) -> list[str]:
        """Get list of available models with pricing."""
        pricing_data = await self.get_current_pricing()
        if pricing_data is None:
            return []

        return pricing_data.model_names()

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache status information."""
        return self.cache.get_cache_info()

    async def clear_cache(self) -> bool:
        """Clear pricing cache."""
        self._current_pricing = None
        return self.cache.clear_cache()
