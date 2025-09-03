"""Cost calculation utilities for token-based pricing (plugin-owned).

These helpers live inside the pricing plugin to avoid coupling core to
pricing logic. They accept an optional PricingService instance for callers
that already have one; otherwise they create a default service on demand.
"""

from __future__ import annotations

from .config import PricingConfig
from .service import PricingService


async def calculate_token_cost(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    pricing_service: PricingService | None = None,
) -> float | None:
    """Calculate total cost in USD for the given token usage.

    If no pricing_service is provided, a default PricingService is created
    using PricingConfig(). Returns None if model or tokens are missing or if
    pricing information is unavailable.
    """
    if not model or (
        not tokens_input
        and not tokens_output
        and not cache_read_tokens
        and not cache_write_tokens
    ):
        return None

    service = pricing_service or PricingService(PricingConfig())

    try:
        cost_decimal = await service.calculate_cost(
            model_name=model,
            input_tokens=tokens_input or 0,
            output_tokens=tokens_output or 0,
            cache_read_tokens=cache_read_tokens or 0,
            cache_write_tokens=cache_write_tokens or 0,
        )
        return float(cost_decimal) if cost_decimal is not None else None
    except Exception:
        return None


async def calculate_cost_breakdown(
    tokens_input: int | None,
    tokens_output: int | None,
    model: str | None,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
    pricing_service: PricingService | None = None,
) -> dict[str, float | str] | None:
    """Return a detailed cost breakdown using current pricing data.

    If no pricing_service is provided, a default PricingService is created.
    Returns None if inputs are insufficient or model pricing is unavailable.
    """
    if not model or (
        not tokens_input
        and not tokens_output
        and not cache_read_tokens
        and not cache_write_tokens
    ):
        return None

    service = pricing_service or PricingService(PricingConfig())

    try:
        model_pricing = await service.get_model_pricing(model)
        if not model_pricing:
            return None

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
    except Exception:
        return None
