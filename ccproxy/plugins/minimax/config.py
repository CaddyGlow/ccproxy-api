"""Configuration for the MiniMax provider plugin."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ccproxy.models.provider import ModelCard, ModelMappingRule, ProviderConfig

from .model_defaults import (
    DEFAULT_MINIMAX_MODEL_CARDS,
    DEFAULT_MINIMAX_MODEL_MAPPINGS,
)


MiniMaxRegion = Literal["global_en", "cn_zh"]


# Per-region upstream endpoints. Selecting a region populates the OpenAI and
# Anthropic base URLs (and documentation root) unless the caller overrides them
# explicitly.
MINIMAX_REGION_ENDPOINTS: dict[str, dict[str, str]] = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
        "docs_root": "https://platform.minimax.io/docs",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
        "docs_root": "https://platform.minimaxi.com/docs",
    },
}


class MiniMaxConfig(ProviderConfig):
    """Provider configuration for the MiniMax API."""

    name: str = "minimax"
    region: MiniMaxRegion = Field(
        default="global_en",
        description=(
            "Upstream region: 'global_en' (minimax.io) or 'cn_zh' (minimaxi.com)."
        ),
    )
    base_url: str = "https://api.minimax.io/v1"
    anthropic_base_url: str = Field(
        default="https://api.minimax.io/anthropic",
        description="Base URL for the MiniMax Anthropic-compatible endpoint.",
    )
    docs_root: str = Field(
        default="https://platform.minimax.io/docs",
        description="Root URL for MiniMax API documentation.",
    )

    supports_streaming: bool = True
    requires_auth: bool = True
    auth_type: str | None = "api_key"

    enabled: bool = True
    priority: int = 5
    default_max_tokens: int = 4096

    api_key: str | None = Field(
        default=None,
        description="MiniMax API key sent as a Bearer token on every upstream request.",
    )
    request_timeout: int = Field(
        default=120,
        description="Timeout for API requests in seconds.",
        ge=1,
        le=600,
    )

    api_headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"},
        description="Default headers for MiniMax API requests.",
    )

    model_mappings: list[ModelMappingRule] = Field(
        default_factory=lambda: [
            rule.model_copy(deep=True) for rule in DEFAULT_MINIMAX_MODEL_MAPPINGS
        ],
        description=(
            "Ordered model translation rules mapping client model identifiers to "
            "MiniMax upstream equivalents."
        ),
    )
    models_endpoint: list[ModelCard] = Field(
        default_factory=lambda: [
            card.model_copy(deep=True) for card in DEFAULT_MINIMAX_MODEL_CARDS
        ],
        description=(
            "Fallback metadata served from /models when the MiniMax API listing is "
            "unavailable."
        ),
    )

    @model_validator(mode="after")
    def _apply_region_endpoints(self) -> MiniMaxConfig:
        """Resolve base URLs from the selected region unless explicitly overridden."""
        endpoints = MINIMAX_REGION_ENDPOINTS.get(self.region)
        if endpoints:
            if "base_url" not in self.model_fields_set:
                self.base_url = endpoints["openai_base_url"]
            if "anthropic_base_url" not in self.model_fields_set:
                self.anthropic_base_url = endpoints["anthropic_base_url"]
            if "docs_root" not in self.model_fields_set:
                self.docs_root = endpoints["docs_root"]
        return self
