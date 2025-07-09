"""Cost calculation utilities for model usage metrics."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ccproxy.config.settings import get_settings
from ccproxy.exceptions import ValidationError


logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    """Pricing information for a specific model."""

    input_tokens_per_1k: float = Field(
        description="Cost per 1K input tokens in USD",
        ge=0,
    )
    output_tokens_per_1k: float = Field(
        description="Cost per 1K output tokens in USD",
        ge=0,
    )
    cache_creation_tokens_per_1k: float = Field(
        default=0.0,
        description="Cost per 1K cache creation tokens in USD",
        ge=0,
    )
    cache_read_tokens_per_1k: float = Field(
        default=0.0,
        description="Cost per 1K cache read tokens in USD",
        ge=0,
    )


class CostCalculator:
    """Calculator for token-based cost calculations."""

    def __init__(self) -> None:
        """Initialize the cost calculator."""
        self.settings = get_settings()
        self._default_pricing = self._get_default_pricing()

    def _get_default_pricing(self) -> dict[str, ModelPricing]:
        """Get default pricing for common models.

        Returns:
            dict: Default pricing configuration for common models
        """
        return {
            # Claude 3.5 Sonnet
            "claude-3-5-sonnet-20241022": ModelPricing(
                input_tokens_per_1k=3.0,
                output_tokens_per_1k=15.0,
                cache_creation_tokens_per_1k=3.75,
                cache_read_tokens_per_1k=0.30,
            ),
            "claude-3-5-sonnet-20240620": ModelPricing(
                input_tokens_per_1k=3.0,
                output_tokens_per_1k=15.0,
                cache_creation_tokens_per_1k=3.75,
                cache_read_tokens_per_1k=0.30,
            ),
            # Claude 3.5 Haiku
            "claude-3-5-haiku-20241022": ModelPricing(
                input_tokens_per_1k=1.0,
                output_tokens_per_1k=5.0,
                cache_creation_tokens_per_1k=1.25,
                cache_read_tokens_per_1k=0.10,
            ),
            # Claude 3 Opus
            "claude-3-opus-20240229": ModelPricing(
                input_tokens_per_1k=15.0,
                output_tokens_per_1k=75.0,
                cache_creation_tokens_per_1k=18.75,
                cache_read_tokens_per_1k=1.50,
            ),
            # Claude 3 Sonnet
            "claude-3-sonnet-20240229": ModelPricing(
                input_tokens_per_1k=3.0,
                output_tokens_per_1k=15.0,
                cache_creation_tokens_per_1k=3.75,
                cache_read_tokens_per_1k=0.30,
            ),
            # Claude 3 Haiku
            "claude-3-haiku-20240307": ModelPricing(
                input_tokens_per_1k=0.25,
                output_tokens_per_1k=1.25,
                cache_creation_tokens_per_1k=0.30,
                cache_read_tokens_per_1k=0.03,
            ),
            # Fallback pricing for unknown models
            "default": ModelPricing(
                input_tokens_per_1k=5.0,
                output_tokens_per_1k=15.0,
                cache_creation_tokens_per_1k=6.25,
                cache_read_tokens_per_1k=0.50,
            ),
        }

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name to handle different naming conventions.

        Args:
            model: Original model name

        Returns:
            str: Normalized model name
        """
        # Remove common prefixes
        normalized = model.lower()

        # Handle OpenAI-style naming
        if normalized.startswith("gpt-"):
            # Map common OpenAI models to Claude equivalents for cost estimation
            # Order matters: more specific patterns first
            openai_mappings = {
                "gpt-4-turbo": "claude-3-5-sonnet-20241022",
                "gpt-4": "claude-3-sonnet-20240229",
                "gpt-3.5-turbo": "claude-3-haiku-20240307",
            }
            for openai_model, claude_model in openai_mappings.items():
                if normalized.startswith(openai_model):
                    logger.debug(
                        f"Mapped OpenAI model {model} to {claude_model} for cost calculation"
                    )
                    return claude_model

        # Handle common Claude model variations
        claude_mappings = {
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku": "claude-3-5-haiku-20241022",
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
        }

        for pattern, full_model in claude_mappings.items():
            if normalized.startswith(pattern):
                logger.debug(f"Normalized model {model} to {full_model}")
                return full_model

        # Return original if no normalization needed
        return model

    def get_model_pricing(self, model: str) -> ModelPricing:
        """Retrieve pricing information for a specific model.

        Args:
            model: Model name to get pricing for

        Returns:
            ModelPricing: Pricing information for the model
        """
        normalized_model = self._normalize_model_name(model)

        # Check if settings has model pricing configuration
        if hasattr(self.settings, "model_pricing") and self.settings.model_pricing:
            settings_pricing = self.settings.model_pricing.get(normalized_model)
            if settings_pricing:
                try:
                    # Convert from settings format to ModelPricing format
                    pricing_data = {
                        "input_tokens_per_1k": settings_pricing.get("input", 0.0),
                        "output_tokens_per_1k": settings_pricing.get("output", 0.0),
                        "cache_creation_tokens_per_1k": settings_pricing.get(
                            "cache_creation", 0.0
                        ),
                        "cache_read_tokens_per_1k": settings_pricing.get(
                            "cache_read", 0.0
                        ),
                    }
                    return ModelPricing(**pricing_data)
                except Exception as e:
                    logger.warning(f"Invalid pricing configuration for {model}: {e}")

        # Use default pricing
        pricing = self._default_pricing.get(normalized_model)
        if pricing:
            return pricing

        # Fallback to default pricing
        logger.warning(f"No pricing found for model {model}, using default pricing")
        return self._default_pricing["default"]

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Calculate the cost for model usage.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_creation_tokens: Number of cache creation tokens (optional)
            cache_read_tokens: Number of cache read tokens (optional)

        Returns:
            float: Total cost in USD

        Raises:
            ValidationError: If token counts are negative
        """
        # Validate token counts
        if input_tokens < 0:
            raise ValidationError(f"Input tokens cannot be negative: {input_tokens}")
        if output_tokens < 0:
            raise ValidationError(f"Output tokens cannot be negative: {output_tokens}")
        if cache_creation_tokens < 0:
            raise ValidationError(
                f"Cache creation tokens cannot be negative: {cache_creation_tokens}"
            )
        if cache_read_tokens < 0:
            raise ValidationError(
                f"Cache read tokens cannot be negative: {cache_read_tokens}"
            )

        # Get pricing for the model
        pricing = self.get_model_pricing(model)

        # Calculate costs (convert from per-1k to per-token)
        input_cost = (input_tokens / 1_000) * pricing.input_tokens_per_1k
        output_cost = (output_tokens / 1_000) * pricing.output_tokens_per_1k
        cache_creation_cost = (
            cache_creation_tokens / 1_000
        ) * pricing.cache_creation_tokens_per_1k
        cache_read_cost = (cache_read_tokens / 1_000) * pricing.cache_read_tokens_per_1k

        total_cost = input_cost + output_cost + cache_creation_cost + cache_read_cost

        logger.debug(
            f"Cost calculation for {model}: "
            f"input={input_tokens}@${pricing.input_tokens_per_1k}/1K=${input_cost:.6f}, "
            f"output={output_tokens}@${pricing.output_tokens_per_1k}/1K=${output_cost:.6f}, "
            f"cache_creation={cache_creation_tokens}@${pricing.cache_creation_tokens_per_1k}/1K=${cache_creation_cost:.6f}, "
            f"cache_read={cache_read_tokens}@${pricing.cache_read_tokens_per_1k}/1K=${cache_read_cost:.6f}, "
            f"total=${total_cost:.6f}"
        )

        return total_cost

    def estimate_streaming_cost(
        self,
        model: str,
        input_tokens: int,
        estimated_output_tokens: int | None = None,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Estimate cost for streaming requests before completion.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            estimated_output_tokens: Estimated number of output tokens (optional)
            cache_creation_tokens: Number of cache creation tokens (optional)
            cache_read_tokens: Number of cache read tokens (optional)

        Returns:
            float: Estimated cost in USD
        """
        # For streaming, we can calculate input and cache costs immediately
        # Output cost estimation is optional
        output_tokens = (
            estimated_output_tokens if estimated_output_tokens is not None else 0
        )

        return self.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )


# Global cost calculator instance
_cost_calculator: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    """Get the global cost calculator instance.

    Returns:
        CostCalculator: The global cost calculator instance
    """
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator
