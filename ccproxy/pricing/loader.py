"""Pricing data loader and format converter for LiteLLM pricing data."""

from decimal import Decimal
from typing import Any

from pydantic import ValidationError
from structlog import get_logger

from .models import PricingData


logger = get_logger(__name__)


class PricingLoader:
    """Loads and converts pricing data from LiteLLM format to internal format."""

    # Claude model name mappings for different versions
    CLAUDE_MODEL_MAPPINGS = {
        # Map versioned models to their canonical names
        "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
        "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-opus-20240229": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    }

    @staticmethod
    def extract_claude_models(
        litellm_data: dict[str, Any], verbose: bool = True
    ) -> dict[str, Any]:
        """Extract Claude model entries from LiteLLM data.

        Args:
            litellm_data: Raw LiteLLM pricing data
            verbose: Whether to log individual model discoveries

        Returns:
            Dictionary with only Claude models
        """
        claude_models = {}

        for model_name, model_data in litellm_data.items():
            # Check if this is a Claude model
            if (
                isinstance(model_data, dict)
                and model_data.get("litellm_provider") == "anthropic"
                and "claude" in model_name.lower()
            ):
                claude_models[model_name] = model_data
                if verbose:
                    logger.debug("claude_model_found", model_name=model_name)

        if verbose:
            logger.info(
                "claude_models_extracted",
                model_count=len(claude_models),
                source="LiteLLM",
            )
        return claude_models

    @staticmethod
    def convert_to_internal_format(
        claude_models: dict[str, Any], verbose: bool = True
    ) -> dict[str, dict[str, Decimal]]:
        """Convert LiteLLM pricing format to internal format.

        LiteLLM format uses cost per token, we use cost per 1M tokens as Decimal.

        Args:
            claude_models: Claude models in LiteLLM format
            verbose: Whether to log individual model conversions

        Returns:
            Dictionary in internal pricing format
        """
        internal_format = {}

        for model_name, model_data in claude_models.items():
            try:
                # Extract pricing fields
                input_cost_per_token = model_data.get("input_cost_per_token")
                output_cost_per_token = model_data.get("output_cost_per_token")
                cache_creation_cost = model_data.get("cache_creation_input_token_cost")
                cache_read_cost = model_data.get("cache_read_input_token_cost")

                # Skip models without pricing info
                if input_cost_per_token is None or output_cost_per_token is None:
                    if verbose:
                        logger.warning("model_pricing_missing", model_name=model_name)
                    continue

                # Convert to per-1M-token pricing (multiply by 1,000,000)
                pricing = {
                    "input": Decimal(str(input_cost_per_token * 1_000_000)),
                    "output": Decimal(str(output_cost_per_token * 1_000_000)),
                }

                # Add cache pricing if available
                if cache_creation_cost is not None:
                    pricing["cache_write"] = Decimal(
                        str(cache_creation_cost * 1_000_000)
                    )

                if cache_read_cost is not None:
                    pricing["cache_read"] = Decimal(str(cache_read_cost * 1_000_000))

                # Map to canonical model name if needed
                canonical_name = PricingLoader.CLAUDE_MODEL_MAPPINGS.get(
                    model_name, model_name
                )
                internal_format[canonical_name] = pricing

                if verbose:
                    logger.debug(
                        "model_pricing_converted",
                        original_name=model_name,
                        canonical_name=canonical_name,
                        input_cost=str(pricing["input"]),
                        output_cost=str(pricing["output"]),
                    )

            except (ValueError, TypeError) as e:
                if verbose:
                    logger.error(
                        "pricing_conversion_failed", model_name=model_name, error=str(e)
                    )
                continue

        if verbose:
            logger.info("models_converted", model_count=len(internal_format))
        return internal_format

    @staticmethod
    def load_pricing_from_data(
        litellm_data: dict[str, Any], verbose: bool = True
    ) -> PricingData | None:
        """Load and convert pricing data from LiteLLM format.

        Args:
            litellm_data: Raw LiteLLM pricing data
            verbose: Whether to enable verbose logging

        Returns:
            Validated pricing data as PricingData model, or None if invalid
        """
        try:
            # Extract Claude models
            claude_models = PricingLoader.extract_claude_models(
                litellm_data, verbose=verbose
            )

            if not claude_models:
                if verbose:
                    logger.warning("claude_models_not_found", source="LiteLLM")
                return None

            # Convert to internal format
            internal_pricing = PricingLoader.convert_to_internal_format(
                claude_models, verbose=verbose
            )

            if not internal_pricing:
                if verbose:
                    logger.warning("pricing_data_invalid")
                return None

            # Validate and create PricingData model
            pricing_data = PricingData.from_dict(internal_pricing)

            if verbose:
                logger.info("pricing_data_loaded", model_count=len(pricing_data))

            return pricing_data

        except ValidationError as e:
            if verbose:
                logger.error("pricing_validation_failed", error=str(e))
            return None
        except Exception as e:
            if verbose:
                logger.error("pricing_load_failed", source="LiteLLM", error=str(e))
            return None

    @staticmethod
    def validate_pricing_data(
        pricing_data: Any, verbose: bool = True
    ) -> PricingData | None:
        """Validate pricing data using Pydantic models.

        Args:
            pricing_data: Pricing data to validate (dict or PricingData)
            verbose: Whether to enable verbose logging

        Returns:
            Valid PricingData model or None if validation fails
        """
        try:
            # If already a PricingData instance, return it
            if isinstance(pricing_data, PricingData):
                if verbose:
                    logger.debug(
                        "pricing_already_validated", model_count=len(pricing_data)
                    )
                return pricing_data

            # If it's a dict, try to create PricingData from it
            if isinstance(pricing_data, dict):
                if not pricing_data:
                    if verbose:
                        logger.warning("pricing_data_empty")
                    return None

                # Try to create PricingData model
                validated_data = PricingData.from_dict(pricing_data)

                if verbose:
                    logger.debug(
                        "pricing_data_validated", model_count=len(validated_data)
                    )

                return validated_data

            # Invalid type
            if verbose:
                logger.error(
                    "pricing_data_invalid_type",
                    actual_type=type(pricing_data).__name__,
                    expected_types=["dict", "PricingData"],
                )
            return None

        except ValidationError as e:
            if verbose:
                logger.error("pricing_validation_failed", error=str(e))
            return None
        except Exception as e:
            if verbose:
                logger.error("pricing_validation_unexpected_error", error=str(e))
            return None

    @staticmethod
    def get_model_aliases() -> dict[str, str]:
        """Get mapping of model aliases to canonical names.

        Returns:
            Dictionary mapping aliases to canonical model names
        """
        return PricingLoader.CLAUDE_MODEL_MAPPINGS.copy()

    @staticmethod
    def get_canonical_model_name(model_name: str) -> str:
        """Get canonical model name for a given model name.

        Args:
            model_name: Model name (possibly an alias)

        Returns:
            Canonical model name
        """
        return PricingLoader.CLAUDE_MODEL_MAPPINGS.get(model_name, model_name)
