"""Claude API plugin configuration."""

from pydantic import Field

from ccproxy.models.provider import ModelCard, ModelMappingRule, ProviderConfig
from ccproxy.plugins.claude_shared.model_defaults import (
    DEFAULT_CLAUDE_MODEL_CARDS,
    DEFAULT_CLAUDE_MODEL_MAPPINGS,
)


class ClaudeAPISettings(ProviderConfig):
    """Claude API specific configuration.

    This configuration extends the base ProviderConfig to include
    Claude API specific settings like API endpoint and model support.
    """

    # Base configuration from ProviderConfig
    name: str = "claude-api"
    base_url: str = "https://api.anthropic.com"
    supports_streaming: bool = True
    requires_auth: bool = True
    auth_type: str = "oauth"

    # Claude API specific settings
    enabled: bool = True
    priority: int = 5  # Higher priority than SDK-based approach
    default_max_tokens: int = 4096

    model_mappings: list[ModelMappingRule] = Field(
        default_factory=lambda: [
            rule.model_copy(deep=True) for rule in DEFAULT_CLAUDE_MODEL_MAPPINGS
        ]
    )
    models_endpoint: list[ModelCard] = Field(
        default_factory=lambda: [
            card.model_copy(deep=True) for card in DEFAULT_CLAUDE_MODEL_CARDS
        ]
    )

    # Feature flags
    include_sdk_content_as_xml: bool = False
    support_openai_format: bool = True  # Support both Anthropic and OpenAI formats

    # System prompt injection mode
    system_prompt_injection_mode: str = "minimal"  # "none", "minimal", or "full"

    # NEW: Auth manager override support
    auth_manager: str | None = (
        None  # Override auth manager name (e.g., 'oauth_claude_lb' for load balancing)
    )
