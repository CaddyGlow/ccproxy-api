"""Service for managing token limits and max_tokens modifications."""

import json
from pathlib import Path
from typing import Any

from ccproxy.core.logging import get_plugin_logger

from .config import MaxTokensConfig
from .models import MaxTokensModification, ModelTokenLimits, TokenLimitsData


logger = get_plugin_logger(__name__)


class TokenLimitsService:
    """Service for managing model token limits and max_tokens modifications."""

    def __init__(self, config: MaxTokensConfig):
        """Initialize token limits service."""
        self.config = config
        self.token_limits_data = TokenLimitsData()
        self._pricing_cache_path = (
            Path.home() / ".cache" / "ccproxy" / "model_pricing.json"
        )

        if self.config.prioritize_local_file:
            # Load local file first (takes precedence)
            self._load_limits_from_local_file()
            self._load_limits_from_pricing_cache()
        else:
            # Load pricing cache first, local file as fallback
            self._load_limits_from_pricing_cache()
            self._load_limits_from_local_file()

    def _load_limits_from_pricing_cache(self) -> None:
        """Load token limits from pricing plugin cache."""
        logger.debug(
            "loading_token_limits_from_pricing_cache",
            cache_path=str(self._pricing_cache_path),
        )

        if not self._pricing_cache_path.exists():
            logger.warning(
                "pricing_cache_not_found_plugin_will_not_modify_requests",
                cache_path=str(self._pricing_cache_path),
                message="max_tokens plugin requires pricing cache to operate",
            )
            return

        try:
            with self._pricing_cache_path.open("r", encoding="utf-8") as f:
                pricing_data = json.load(f)

            loaded_count = 0
            for model_name, model_data in pricing_data.items():
                # Skip non-dict entries (like headers or metadata)
                if not isinstance(model_data, dict):
                    continue

                # Skip image generation models and other non-text models
                if model_data.get("mode") == "image_generation":
                    continue

                # Extract max_output_tokens (prefer this over max_tokens)
                max_output = model_data.get("max_output_tokens") or model_data.get(
                    "max_tokens"
                )
                max_input = model_data.get("max_input_tokens")

                # Skip if values are not integers (e.g., documentation strings)
                if not isinstance(max_output, int):
                    continue

                if max_output:
                    self.token_limits_data.models[model_name] = ModelTokenLimits(
                        max_output_tokens=max_output,
                        max_input_tokens=max_input
                        if isinstance(max_input, int)
                        else None,
                    )
                    loaded_count += 1

            logger.debug(
                "token_limits_loaded_from_pricing_cache",
                model_count=loaded_count,
                cache_path=str(self._pricing_cache_path),
            )

        except Exception as e:
            logger.error(
                "failed_to_load_pricing_cache_plugin_will_not_modify_requests",
                cache_path=str(self._pricing_cache_path),
                error=str(e),
                exc_info=e,
            )

    def _load_limits_from_local_file(self) -> None:
        """Load token limits from local token_limits.json file."""
        local_file_path = Path(self.config.default_token_limits_file)

        logger.debug(
            "loading_token_limits_from_local_file",
            file_path=str(local_file_path),
        )

        if not local_file_path.exists():
            logger.debug(
                "local_token_limits_file_not_found",
                file_path=str(local_file_path),
            )
            return

        try:
            with local_file_path.open("r", encoding="utf-8") as f:
                local_data = json.load(f)

            # Handle flat structure like pricing cache (no nested "models" object)
            models_data = {}
            if "models" in local_data:
                # Old format with nested models
                models_data = local_data.get("models", {})
                if not isinstance(models_data, dict):
                    logger.warning(
                        "invalid_local_token_limits_format",
                        file_path=str(local_file_path),
                        reason="models section is not a dictionary",
                    )
                    return
            else:
                # New flat format like pricing cache
                models_data = {
                    k: v
                    for k, v in local_data.items()
                    if not k.startswith("_") and isinstance(v, dict)
                }

            loaded_count = 0
            for model_name, model_limits in models_data.items():
                if not isinstance(model_limits, dict):
                    continue

                max_output = model_limits.get("max_output_tokens")
                max_input = model_limits.get("max_input_tokens")

                if isinstance(max_output, int) and max_output > 0:
                    if self.config.prioritize_local_file:
                        # Local file values take precedence over pricing cache
                        self.token_limits_data.models[model_name] = ModelTokenLimits(
                            max_output_tokens=max_output,
                            max_input_tokens=max_input
                            if isinstance(max_input, int) and max_input > 0
                            else None,
                        )
                        loaded_count += 1
                    else:
                        # Local file is fallback - only add if model doesn't exist
                        if model_name not in self.token_limits_data.models:
                            self.token_limits_data.models[model_name] = (
                                ModelTokenLimits(
                                    max_output_tokens=max_output,
                                    max_input_tokens=max_input
                                    if isinstance(max_input, int) and max_input > 0
                                    else None,
                                )
                            )
                            loaded_count += 1

            logger.debug(
                "token_limits_loaded_from_local_file",
                file_path=str(local_file_path),
                model_count=loaded_count,
            )

        except Exception as e:
            logger.error(
                "failed_to_load_local_token_limits_file",
                file_path=str(local_file_path),
                error=str(e),
                exc_info=e,
            )

    def get_max_output_tokens(self, model_name: str) -> int | None:
        """Get maximum output tokens for a model."""
        return self.token_limits_data.get_max_output_tokens(model_name)

    def should_modify_max_tokens(
        self, request_data: dict[str, Any], model: str
    ) -> tuple[bool, str]:
        """Determine if max_tokens should be modified for the request."""
        current_max_tokens = request_data.get("max_tokens")

        # Enforce mode: always modify to set max_tokens to model limit
        if self.config.enforce_mode:
            return True, "enforced"

        # Case 1: No max_tokens provided
        if current_max_tokens is None:
            return True, "missing"

        # Case 2: Invalid max_tokens (not a positive integer)
        if not isinstance(current_max_tokens, int) or current_max_tokens <= 0:
            return True, "invalid"

        # Case 3: Max tokens exceeds model limit
        model_limit = self.get_max_output_tokens(model)
        if model_limit and current_max_tokens > model_limit:
            return True, "exceeded"

        # No modification needed
        return False, "none"

    def modify_max_tokens(
        self, request_data: dict[str, Any], model: str, provider: str | None = None
    ) -> tuple[dict[str, Any], MaxTokensModification | None]:
        """Modify max_tokens in request data if needed."""
        should_modify, reason_type = self.should_modify_max_tokens(request_data, model)

        if not should_modify:
            return request_data, None

        original_max_tokens = request_data.get("max_tokens")

        # Determine the appropriate max_tokens value
        model_limit = self.get_max_output_tokens(model)

        if model_limit:
            new_max_tokens = model_limit
        else:
            # Use fallback when model limit is unknown
            new_max_tokens = self.config.fallback_max_tokens
            logger.debug(
                "using_fallback_max_tokens",
                model=model,
                fallback=self.config.fallback_max_tokens,
            )

        # Create modification info
        modification = MaxTokensModification(
            original_max_tokens=original_max_tokens,
            new_max_tokens=new_max_tokens,
            model=model,
            reason=self.config.get_modification_reason(reason_type),
        )

        # Create modified request data
        modified_data = request_data.copy()
        modified_data["max_tokens"] = new_max_tokens

        if self.config.log_modifications:
            logger.info(
                "max_tokens_modified",
                model=model,
                provider=provider,
                original=original_max_tokens,
                new=new_max_tokens,
                reason=modification.reason,
            )

        return modified_data, modification

    def initialize(self) -> None:
        """Initialize the service."""
        logger.debug(
            "token_limits_service_initialized",
            models_count=len(self.token_limits_data.models),
            pricing_cache=str(self._pricing_cache_path),
            fallback=self.config.fallback_max_tokens,
            enforce_mode=self.config.enforce_mode,
            prioritize_local_file=self.config.prioritize_local_file,
        )
