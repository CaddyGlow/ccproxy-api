"""Pricing data loader and format converter for LiteLLM pricing data."""

import json
from decimal import Decimal
from typing import Any, Literal

import httpx
from pydantic import ValidationError

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins.claude_shared.model_defaults import (
    DEFAULT_CLAUDE_MODEL_MAPPINGS,
)
from ccproxy.utils.model_mapper import ModelMapper

from .models import PricingData


logger = get_plugin_logger(__name__)

_CLAUDE_MODEL_MAPPER = ModelMapper(DEFAULT_CLAUDE_MODEL_MAPPINGS)
_CLAUDE_ALIAS_MAP: dict[str, str] = {
    rule.match: rule.target
    for rule in DEFAULT_CLAUDE_MODEL_MAPPINGS
    if rule.match.startswith("claude-")
}


def _is_openai_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return lowered.startswith(("gpt-", "o1", "o3", "text-davinci"))


class PricingLoader:
    """Loads and converts pricing data from LiteLLM format to internal format."""

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
    def extract_openai_models(
        litellm_data: dict[str, Any], verbose: bool = True
    ) -> dict[str, Any]:
        """Extract OpenAI model entries from LiteLLM data.

        Args:
            litellm_data: Raw LiteLLM pricing data
            verbose: Whether to log individual model discoveries

        Returns:
            Dictionary with only OpenAI models
        """
        openai_models = {}

        for model_name, model_data in litellm_data.items():
            # Check if this is an OpenAI model
            if isinstance(model_data, dict) and (
                model_data.get("litellm_provider") == "openai"
                or _is_openai_model(model_name)
            ):
                openai_models[model_name] = model_data
                if verbose:
                    logger.debug("openai_model_found", model_name=model_name)

        if verbose:
            logger.info(
                "openai_models_extracted",
                model_count=len(openai_models),
                source="LiteLLM",
            )
        return openai_models

    @staticmethod
    def extract_anthropic_models(
        litellm_data: dict[str, Any], verbose: bool = True
    ) -> dict[str, Any]:
        """Extract all Anthropic model entries from LiteLLM data.

        This includes Claude models and any other Anthropic models.

        Args:
            litellm_data: Raw LiteLLM pricing data
            verbose: Whether to log individual model discoveries

        Returns:
            Dictionary with all Anthropic models
        """
        anthropic_models = {}

        for model_name, model_data in litellm_data.items():
            # Check if this is an Anthropic model
            if (
                isinstance(model_data, dict)
                and model_data.get("litellm_provider") == "anthropic"
            ):
                anthropic_models[model_name] = model_data
                if verbose:
                    logger.debug("anthropic_model_found", model_name=model_name)

        if verbose:
            logger.info(
                "anthropic_models_extracted",
                model_count=len(anthropic_models),
                source="LiteLLM",
            )
        return anthropic_models

    @staticmethod
    def extract_models_by_provider(
        litellm_data: dict[str, Any],
        provider: Literal["anthropic", "openai", "all", "claude"] = "all",
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Extract models by provider from LiteLLM data.

        Args:
            litellm_data: Raw LiteLLM pricing data
            provider: Provider to extract models for ("anthropic", "openai", "claude", or "all")
            verbose: Whether to log individual model discoveries

        Returns:
            Dictionary with models from specified provider(s)
        """
        if provider == "claude":
            return PricingLoader.extract_claude_models(litellm_data, verbose)
        elif provider == "anthropic":
            return PricingLoader.extract_anthropic_models(litellm_data, verbose)
        elif provider == "openai":
            return PricingLoader.extract_openai_models(litellm_data, verbose)
        elif provider == "all":
            # Extract all models that have pricing data
            all_models = {}
            for model_name, model_data in litellm_data.items():
                if isinstance(model_data, dict):
                    all_models[model_name] = model_data
                    if verbose:
                        provider_name = model_data.get("litellm_provider", "unknown")
                        logger.debug(
                            "model_found",
                            model_name=model_name,
                            provider=provider_name,
                        )

            if verbose:
                logger.info(
                    "all_models_extracted",
                    model_count=len(all_models),
                    source="LiteLLM",
                )
            return all_models
        else:
            raise ValueError(
                f"Invalid provider: {provider}. Use 'anthropic', 'openai', 'claude', or 'all'"
            )

    @staticmethod
    def convert_to_internal_format(
        models: dict[str, Any], map_to_claude: bool = True, verbose: bool = True
    ) -> dict[str, dict[str, Decimal]]:
        """Convert LiteLLM pricing format to internal format.

        LiteLLM format uses cost per token, we use cost per 1M tokens as Decimal.

        Args:
            models: Models in LiteLLM format
            map_to_claude: Whether to map model names to Claude equivalents
            verbose: Whether to log individual model conversions

        Returns:
            Dictionary in internal pricing format
        """
        internal_format = {}

        for model_name, model_data in models.items():
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

                # Optionally map to canonical model name
                if map_to_claude:
                    canonical_name = _CLAUDE_MODEL_MAPPER.map(model_name).mapped
                else:
                    canonical_name = model_name

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
        litellm_data: dict[str, Any],
        provider: Literal["anthropic", "openai", "all", "claude"] = "claude",
        map_to_claude: bool = True,
        verbose: bool = True,
    ) -> PricingData | None:
        """Load and convert pricing data from LiteLLM format.

        Args:
            litellm_data: Raw LiteLLM pricing data
            provider: Provider to load pricing for ("anthropic", "openai", "all", or "claude")
                     "claude" is kept for backward compatibility and extracts only Claude models
            map_to_claude: Whether to map model names to Claude equivalents
            verbose: Whether to enable verbose logging

        Returns:
            Validated pricing data as PricingData model, or None if invalid
        """
        try:
            # Extract models based on provider
            if provider == "claude":
                # Backward compatibility - extract only Claude models
                models = PricingLoader.extract_claude_models(
                    litellm_data, verbose=verbose
                )
            else:
                models = PricingLoader.extract_models_by_provider(
                    litellm_data, provider=provider, verbose=verbose
                )

            if not models:
                if verbose:
                    logger.warning(
                        "models_not_found", provider=provider, source="LiteLLM"
                    )
                return None

            # Convert to internal format
            internal_pricing = PricingLoader.convert_to_internal_format(
                models, map_to_claude=map_to_claude, verbose=verbose
            )

            if not internal_pricing:
                if verbose:
                    logger.warning("pricing_data_invalid")
                return None

            # Validate and create PricingData model
            pricing_data = PricingData.model_validate(internal_pricing)

            if verbose:
                logger.info(
                    "pricing_data_loaded",
                    model_count=len(pricing_data),
                    provider=provider,
                )

            return pricing_data

        except ValidationError as e:
            if verbose:
                logger.error("pricing_validation_failed", error=str(e), exc_info=e)
            return None
        except json.JSONDecodeError as e:
            if verbose:
                logger.error(
                    "pricing_json_decode_failed",
                    source="LiteLLM",
                    error=str(e),
                    exc_info=e,
                )
            return None
        except httpx.HTTPError as e:
            if verbose:
                logger.error(
                    "pricing_http_error", source="LiteLLM", error=str(e), exc_info=e
                )
            return None
        except OSError as e:
            if verbose:
                logger.error(
                    "pricing_io_error", source="LiteLLM", error=str(e), exc_info=e
                )
            return None
        except Exception as e:
            if verbose:
                logger.error(
                    "pricing_load_failed", source="LiteLLM", error=str(e), exc_info=e
                )
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
                validated_data = PricingData.model_validate(pricing_data)

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
                logger.error("pricing_validation_failed", error=str(e), exc_info=e)
            return None
        except json.JSONDecodeError as e:
            if verbose:
                logger.error("pricing_validation_json_error", error=str(e), exc_info=e)
            return None
        except OSError as e:
            if verbose:
                logger.error("pricing_validation_io_error", error=str(e), exc_info=e)
            return None
        except Exception as e:
            if verbose:
                logger.error(
                    "pricing_validation_unexpected_error", error=str(e), exc_info=e
                )
            return None

    @staticmethod
    def get_model_aliases() -> dict[str, str]:
        """Get mapping of model aliases to canonical names.

        Returns:
            Dictionary mapping aliases to canonical model names
        """
        return _CLAUDE_ALIAS_MAP.copy()

    @staticmethod
    def get_canonical_model_name(model_name: str) -> str:
        """Get canonical model name for a given model name.

        Args:
            model_name: Model name (possibly an alias)

        Returns:
            Canonical model name
        """
        return _CLAUDE_MODEL_MAPPER.map(model_name).mapped
