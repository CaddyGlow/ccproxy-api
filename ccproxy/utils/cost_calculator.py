"""Cost calculation utilities for token-based pricing.

This module provides shared cost calculation functionality that can be used
across different services to ensure consistent pricing calculations.
The pricing functionality is optional and depends on the pricing plugin.
"""

from typing import TYPE_CHECKING

from ccproxy.core.logging import get_logger


if TYPE_CHECKING:
    from plugins.pricing.service import PricingService


logger = get_logger(__name__)


async def calculate_token_cost(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    pricing_service: "PricingService | None" = None,
) -> float | None:
    """Calculate cost in USD for the given token usage including cache tokens.

    This is a shared utility function that provides consistent cost calculation
    across all services using the pricing system when available.

    Args:
        tokens_input: Number of input tokens
        tokens_output: Number of output tokens
        model: Model name for pricing lookup
        cache_read_tokens: Number of cache read tokens
        cache_write_tokens: Number of cache write tokens
        pricing_service: Pricing service instance (optional)

    Returns:
        Cost in USD or None if calculation not possible
    """
    logger = get_logger(__name__)

    if not model or (
        not tokens_input
        and not tokens_output
        and not cache_read_tokens
        and not cache_write_tokens
    ):
        return None

    try:
        # Check if pricing service is provided
        if not pricing_service:
            logger.debug(
                "cost_calculation_skipped", reason="pricing_service_not_provided"
            )
            return None

        # Calculate cost using pricing service
        cost_decimal = await pricing_service.calculate_cost(
            model_name=model,
            input_tokens=tokens_input or 0,
            output_tokens=tokens_output or 0,
            cache_read_tokens=cache_read_tokens or 0,
            cache_write_tokens=cache_write_tokens or 0,
        )

        if cost_decimal is None:
            logger.debug(
                "cost_calculation_skipped",
                model=model,
                reason="model_not_found_or_pricing_unavailable",
            )
            return None

        total_cost = float(cost_decimal)

        logger.debug(
            "cost_calculated",
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=total_cost,
        )

        return total_cost

    except Exception as e:
        logger.debug("cost_calculation_error", error=str(e), model=model, exc_info=e)
        return None


async def calculate_cost_breakdown(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    pricing_service: "PricingService | None" = None,
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
        # Check if pricing service is provided
        if not pricing_service:
            return None

        # Get model pricing
        model_pricing = await pricing_service.get_model_pricing(model)
        if not model_pricing:
            return None

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
            "model": model,
        }

    except Exception as e:
        logger.debug("cost_breakdown_error", error=str(e), model=model, exc_info=e)
        return None
