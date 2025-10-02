"""Configuration for max_tokens plugin."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MaxTokensConfig(BaseModel):
    """Configuration for the max_tokens plugin."""

    enabled: bool = Field(
        default=True, description="Whether the max_tokens plugin is enabled"
    )
    default_token_limits_file: str = Field(
        default=str(Path(__file__).parent / "token_limits.json"),
        description="Path to JSON file containing default token limits",
    )
    fallback_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Fallback max_tokens when model limits are unknown",
    )
    apply_to_all_providers: bool = Field(
        default=True,
        description="Whether to apply to all providers or only specific ones",
    )
    target_providers: list[str] = Field(
        default_factory=lambda: ["claude_api", "claude_sdk", "codex", "copilot"],
        description="List of providers to apply max_tokens modifications to",
    )
    require_pricing_data: bool = Field(
        default=False,
        description=(
            "If True, only modify requests when pricing data is available. "
            "If False, use fallback limits when pricing data is not available."
        ),
    )
    log_modifications: bool = Field(
        default=True, description="Whether to log max_tokens modifications"
    )
    enforce_mode: bool = Field(
        default=False,
        description=(
            "When enabled, always set max_tokens to the model's maximum limit, "
            "ignoring the request's current max_tokens value"
        ),
    )
    prioritize_local_file: bool = Field(
        default=False,
        description=(
            "When enabled, local token_limits.json values take precedence over "
            "pricing cache values. When disabled, local file is only used as fallback "
            "when pricing cache is unavailable or model is not found in cache."
        ),
    )
    modification_reasons: dict[str, str] = Field(
        default_factory=lambda: {
            "missing": "max_tokens was missing from request",
            "invalid": "max_tokens was invalid or too high",
            "exceeded": "max_tokens exceeded model limit",
            "enforced": "max_tokens enforced to model limit (enforce mode)",
        },
        description="Reason templates for modifications",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration values."""
        # Ensure target_providers is a list
        if isinstance(data.get("target_providers"), str):
            data["target_providers"] = [data["target_providers"]]
        return data

    def should_process_provider(self, provider: str) -> bool:
        """Check if plugin should process requests for given provider."""
        if self.apply_to_all_providers:
            return True
        return provider in self.target_providers

    def get_modification_reason(self, reason_type: str) -> str:
        """Get modification reason text for given reason type."""
        return self.modification_reasons.get(
            reason_type, f"Unknown reason: {reason_type}"
        )
