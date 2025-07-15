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
                    logger.debug(f"Found Claude model: {model_name}")

        if verbose:
            logger.info(
                f"Extracted {len(claude_models)} Claude models from LiteLLM data"
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
                        logger.warning(
                            f"Model {model_name} missing required pricing fields"
                        )
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
                        f"Converted {model_name} -> {canonical_name}: "
                        f"input=${pricing['input']}, output=${pricing['output']}"
                    )

            except (ValueError, TypeError) as e:
                if verbose:
                    logger.error(f"Failed to convert pricing for {model_name}: {e}")
                continue

        if verbose:
            logger.info(f"Converted {len(internal_format)} models to internal format")
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
                    logger.warning("No Claude models found in LiteLLM data")
                return None

            # Convert to internal format
            internal_pricing = PricingLoader.convert_to_internal_format(
                claude_models, verbose=verbose
            )

            if not internal_pricing:
                if verbose:
                    logger.warning("No valid pricing data after conversion")
                return None

            # Validate and create PricingData model
            pricing_data = PricingData.from_dict(internal_pricing)

            if verbose:
                logger.info(
                    f"Successfully loaded pricing for {len(pricing_data)} models"
                )

            return pricing_data

        except ValidationError as e:
            if verbose:
                logger.error(f"Pricing data validation failed: {e}")
            return None
        except Exception as e:
            if verbose:
                logger.error(f"Failed to load pricing from LiteLLM data: {e}")
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
                        f"Pricing data already validated for {len(pricing_data)} models"
                    )
                return pricing_data

            # If it's a dict, try to create PricingData from it
            if isinstance(pricing_data, dict):
                if not pricing_data:
                    if verbose:
                        logger.warning("Pricing data is empty")
                    return None

                # Try to create PricingData model
                validated_data = PricingData.from_dict(pricing_data)

                if verbose:
                    logger.debug(
                        f"Validated pricing data for {len(validated_data)} models"
                    )

                return validated_data

            # Invalid type
            if verbose:
                logger.error(
                    f"Pricing data must be dict or PricingData, got {type(pricing_data)}"
                )
            return None

        except ValidationError as e:
            if verbose:
                logger.error(f"Pricing data validation failed: {e}")
            return None
        except Exception as e:
            if verbose:
                logger.error(f"Unexpected error validating pricing data: {e}")
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
