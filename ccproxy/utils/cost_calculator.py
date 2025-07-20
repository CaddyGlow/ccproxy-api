"""Cost calculation utilities for token-based pricing.

This module provides shared cost calculation functionality that can be used
across different services to ensure consistent pricing calculations.
"""

import structlog


logger = structlog.get_logger(__name__)


def calculate_token_cost(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> float | None:
    """Calculate cost in USD for the given token usage including cache tokens.

    This is a shared utility function that provides consistent cost calculation
    across all services using the pricing data from the pricing system.

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        model: Model name for pricing lookup
        cache_read_tokens: Number of cache read tokens
        cache_write_tokens: Number of cache write tokens

    Returns:
        Cost in USD or None if calculation not possible
    """
    if not model or (
        not tokens_input
        and not tokens_output
        and not cache_read_tokens
        and not cache_write_tokens
    ):
        return None

    try:
        # Import pricing system components
        from ccproxy.config.pricing import PricingSettings
        from ccproxy.pricing.cache import PricingCache
        from ccproxy.pricing.loader import PricingLoader

        # Get canonical model name
        canonical_model = PricingLoader.get_canonical_model_name(model)

        # Create pricing components with dependency injection
        settings = PricingSettings()
        cache = PricingCache(settings)
        cached_data = cache.load_cached_data()

        # If cache is expired, try to use stale cache as fallback
        if not cached_data:
            try:
                import json

                if cache.cache_file.exists():
                    with cache.cache_file.open(encoding="utf-8") as f:
                        cached_data = json.load(f)
                    logger.debug(
                        "cost_calculation_using_stale_cache",
                        cache_age_hours=cache.get_cache_info().get("age_hours"),
                    )
            except (OSError, json.JSONDecodeError):
                pass

        if not cached_data:
            logger.debug("cost_calculation_skipped", reason="no_pricing_data")
            return None

        # Load pricing data
        pricing_data = PricingLoader.load_pricing_from_data(cached_data, verbose=False)
        if not pricing_data or canonical_model not in pricing_data:
            logger.debug(
                "cost_calculation_skipped",
                model=canonical_model,
                reason="model_not_found",
            )
            return None

        model_pricing = pricing_data[canonical_model]

        # Calculate cost (pricing is per 1M tokens)
        input_cost = ((tokens_input or 0) / 1_000_000) * float(model_pricing.input)
        output_cost = ((tokens_output or 0) / 1_000_000) * float(model_pricing.output)
        cache_read_cost = ((cache_read_tokens or 0) / 1_000_000) * float(
            model_pricing.cache_read
        )
        cache_write_cost = ((cache_write_tokens or 0) / 1_000_000) * float(
            model_pricing.cache_write
        )

        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

        logger.debug(
            "cost_calculated",
            model=canonical_model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            cost_usd=total_cost,
        )

        return total_cost

    except Exception as e:
        logger.debug("cost_calculation_error", error=str(e), model=model)
        return None


def calculate_cost_breakdown(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> dict[str, float | str] | None:
    """Calculate detailed cost breakdown for the given token usage.

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        model: Model name for pricing lookup
        cache_read_tokens: Number of cache read tokens
        cache_write_tokens: Number of cache write tokens

    Returns:
        Dictionary with cost breakdown or None if calculation not possible
    """
    if not model or (
        not tokens_input
        and not tokens_output
        and not cache_read_tokens
        and not cache_write_tokens
    ):
        return None

    try:
        # Import pricing system components
        from ccproxy.config.pricing import PricingSettings
        from ccproxy.pricing.cache import PricingCache
        from ccproxy.pricing.loader import PricingLoader

        # Get canonical model name
        canonical_model = PricingLoader.get_canonical_model_name(model)

        # Create pricing components with dependency injection
        settings = PricingSettings()
        cache = PricingCache(settings)
        cached_data = cache.load_cached_data()

        # If cache is expired, try to use stale cache as fallback
        if not cached_data:
            try:
                import json

                if cache.cache_file.exists():
                    with cache.cache_file.open(encoding="utf-8") as f:
                        cached_data = json.load(f)
                    logger.debug(
                        "cost_breakdown_using_stale_cache",
                        cache_age_hours=cache.get_cache_info().get("age_hours"),
                    )
            except (OSError, json.JSONDecodeError):
                pass

        if not cached_data:
            return None

        # Load pricing data
        pricing_data = PricingLoader.load_pricing_from_data(cached_data, verbose=False)
        if not pricing_data or canonical_model not in pricing_data:
            return None

        model_pricing = pricing_data[canonical_model]

        # Calculate individual costs (pricing is per 1M tokens)
        input_cost = ((tokens_input or 0) / 1_000_000) * float(model_pricing.input)
        output_cost = ((tokens_output or 0) / 1_000_000) * float(model_pricing.output)
        cache_read_cost = ((cache_read_tokens or 0) / 1_000_000) * float(
            model_pricing.cache_read
        )
        cache_write_cost = ((cache_write_tokens or 0) / 1_000_000) * float(
            model_pricing.cache_write
        )

        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cache_read_cost": cache_read_cost,
            "cache_write_cost": cache_write_cost,
            "total_cost": total_cost,
            "model": canonical_model,
        }

    except Exception as e:
        logger.debug("cost_breakdown_error", error=str(e), model=model)
        return None
