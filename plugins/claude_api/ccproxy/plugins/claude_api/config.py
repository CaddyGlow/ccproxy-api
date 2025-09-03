"""Claude API plugin configuration."""

from ccproxy.models.provider import ProviderConfig


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

    # Supported models
    models: list[str] = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    # Feature flags
    include_sdk_content_as_xml: bool = False
    support_openai_format: bool = True  # Support both Anthropic and OpenAI formats

    # System prompt injection mode
    system_prompt_injection_mode: str = "minimal"  # "none", "minimal", or "full"
