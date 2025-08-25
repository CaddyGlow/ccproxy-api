"""Configuration for Claude SDK plugin."""

from enum import Enum

from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel, Field, model_validator

from ccproxy.models.provider import ProviderConfig


def _create_default_claude_code_options(
    builtin_permissions: bool = True,
    continue_conversation: bool = False,
) -> ClaudeCodeOptions:
    """Create ClaudeCodeOptions with default values.

    Args:
        builtin_permissions: Whether to include built-in permission handling defaults
    """
    if builtin_permissions:
        return ClaudeCodeOptions(
            continue_conversation=continue_conversation,
            mcp_servers={
                "confirmation": {"type": "sse", "url": "http://127.0.0.1:8000/mcp"}
            },
            permission_prompt_tool_name="mcp__confirmation__check_permission",
        )
    else:
        return ClaudeCodeOptions(
            mcp_servers={},
            permission_prompt_tool_name=None,
            continue_conversation=continue_conversation,
        )


class SDKMessageMode(str, Enum):
    """Modes for handling SDK messages from Claude SDK.

    - forward: Forward SDK content blocks directly with original types and metadata
    - ignore: Skip SDK messages and blocks completely
    - formatted: Format as XML tags with JSON data in text deltas
    """

    FORWARD = "forward"
    IGNORE = "ignore"
    FORMATTED = "formatted"


class SystemPromptInjectionMode(str, Enum):
    """Modes for system prompt injection.

    - minimal: Only inject Claude Code identification prompt
    - full: Inject all detected system messages from Claude CLI
    """

    MINIMAL = "minimal"
    FULL = "full"


class SessionPoolSettings(BaseModel):
    """Session pool configuration settings."""

    enabled: bool = Field(
        default=True, description="Enable session-aware persistent pooling"
    )

    session_ttl: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Session time-to-live in seconds (1 minute to 24 hours)",
    )

    max_sessions: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum number of concurrent sessions",
    )

    cleanup_interval: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Session cleanup interval in seconds (30 seconds to 1 hour)",
    )

    idle_threshold: int = Field(
        default=600,
        ge=60,
        le=7200,
        description="Session idle threshold in seconds (1 minute to 2 hours)",
    )

    connection_recovery: bool = Field(
        default=True,
        description="Enable automatic connection recovery for unhealthy sessions",
    )

    stream_first_chunk_timeout: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Stream first chunk timeout in seconds (1-30 seconds)",
    )

    stream_ongoing_timeout: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Stream ongoing timeout in seconds after first chunk (10 seconds to 10 minutes)",
    )

    stream_interrupt_timeout: int = Field(
        default=10,
        ge=2,
        le=60,
        description="Stream interrupt timeout in seconds for SDK and worker operations (2-60 seconds)",
    )

    @model_validator(mode="after")
    def validate_timeout_hierarchy(self) -> "SessionPoolSettings":
        """Ensure stream timeouts are less than session TTL."""
        if self.stream_ongoing_timeout >= self.session_ttl:
            raise ValueError(
                f"stream_ongoing_timeout ({self.stream_ongoing_timeout}s) must be less than session_ttl ({self.session_ttl}s)"
            )

        if self.stream_first_chunk_timeout >= self.stream_ongoing_timeout:
            raise ValueError(
                f"stream_first_chunk_timeout ({self.stream_first_chunk_timeout}s) must be less than stream_ongoing_timeout ({self.stream_ongoing_timeout}s)"
            )

        return self


class ClaudeSDKSettings(ProviderConfig):
    """Claude SDK specific configuration."""

    # Base required fields for ProviderConfig
    name: str = "claude_sdk"
    base_url: str = "claude-sdk://local"  # Special URL for SDK
    supports_streaming: bool = True
    requires_auth: bool = False  # SDK handles auth internally
    auth_type: str | None = None
    models: list[str] = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    # Plugin lifecycle settings
    enabled: bool = True
    priority: int = 0

    # Claude SDK specific settings
    cli_path: str | None = None
    builtin_permissions: bool = True
    session_pool_enabled: bool = False
    session_pool_size: int = 5
    session_timeout_seconds: int = 300

    # SDK behavior settings
    include_system_messages_in_stream: bool = True
    pretty_format: bool = True
    sdk_message_mode: SDKMessageMode = SDKMessageMode.FORWARD

    # Performance settings
    max_tokens_default: int = 4096
    temperature_default: float = 0.7

    # Additional fields from ClaudeSettings to prevent validation errors
    code_options: ClaudeCodeOptions | None = None
    system_prompt_injection_mode: SystemPromptInjectionMode = (
        SystemPromptInjectionMode.MINIMAL
    )
    sdk_session_pool: SessionPoolSettings | None = None

    # Default session configuration
    default_session_id: str | None = Field(
        default=None,
        description="Default session ID to use when none is provided. "
        "Useful for single-user setups or development environments.",
    )
    auto_generate_default_session: bool = Field(
        default=False,
        description="Automatically generate a random default session ID at startup. "
        "Overrides default_session_id if enabled. Useful for single-user "
        "setups where you want session persistence during runtime.",
    )

    @model_validator(mode="after")
    def ensure_session_pool_settings(self) -> "ClaudeSDKSettings":
        """Ensure sdk_session_pool is initialized."""
        if self.sdk_session_pool is None:
            self.sdk_session_pool = SessionPoolSettings()
        return self

    class Config:
        """Pydantic configuration."""

        extra = "allow"  # Allow extra fields from Claude configuration
