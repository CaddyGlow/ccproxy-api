"""Detection models for Claude Code CLI headers and system prompt extraction."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


class ClaudeCodeHeaders(BaseModel):
    """Pydantic model for Claude CLI headers extraction with field aliases."""

    anthropic_beta: str = Field(
        alias="anthropic-beta", description="Anthropic beta features"
    )
    anthropic_version: str = Field(
        alias="anthropic-version", description="Anthropic API version"
    )
    anthropic_dangerous_direct_browser_access: str = Field(
        alias="anthropic-dangerous-direct-browser-access",
        description="Browser access flag",
    )
    x_app: str = Field(alias="x-app", description="Application identifier")
    user_agent: str = Field(alias="user-agent", description="User agent string")
    x_stainless_lang: str = Field(alias="x-stainless-lang", description="SDK language")
    x_stainless_retry_count: str = Field(
        alias="x-stainless-retry-count", description="Retry count"
    )
    x_stainless_timeout: str = Field(
        alias="x-stainless-timeout", description="Request timeout"
    )
    x_stainless_package_version: str = Field(
        alias="x-stainless-package-version", description="Package version"
    )
    x_stainless_os: str = Field(alias="x-stainless-os", description="Operating system")
    x_stainless_arch: str = Field(alias="x-stainless-arch", description="Architecture")
    x_stainless_runtime: str = Field(alias="x-stainless-runtime", description="Runtime")
    x_stainless_runtime_version: str = Field(
        alias="x-stainless-runtime-version", description="Runtime version"
    )

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def to_headers_dict(self) -> dict[str, str]:
        """Convert to headers dictionary for HTTP forwarding with proper case."""
        headers = {}

        # Map field names to proper HTTP header names
        header_mapping = {
            "anthropic_beta": "anthropic-beta",
            "anthropic_version": "anthropic-version",
            "anthropic_dangerous_direct_browser_access": "anthropic-dangerous-direct-browser-access",
            "x_app": "x-app",
            "user_agent": "User-Agent",
            "x_stainless_lang": "X-Stainless-Lang",
            "x_stainless_retry_count": "X-Stainless-Retry-Count",
            "x_stainless_timeout": "X-Stainless-Timeout",
            "x_stainless_package_version": "X-Stainless-Package-Version",
            "x_stainless_os": "X-Stainless-OS",
            "x_stainless_arch": "X-Stainless-Arch",
            "x_stainless_runtime": "X-Stainless-Runtime",
            "x_stainless_runtime_version": "X-Stainless-Runtime-Version",
        }

        for field_name, header_name in header_mapping.items():
            value = getattr(self, field_name, None)
            if value is not None:
                headers[header_name] = value

        return headers


class SystemPromptData(BaseModel):
    """Extracted system prompt information."""

    text: Annotated[str, Field(description="System prompt text content")]
    cache_control: Annotated[
        dict[str, Any] | None, Field(description="Cache control settings")
    ] = None

    model_config = ConfigDict(extra="forbid")


class ClaudeCacheData(BaseModel):
    """Cached Claude CLI detection data with version tracking."""

    claude_version: Annotated[str, Field(description="Claude CLI version")]
    headers: Annotated[ClaudeCodeHeaders, Field(description="Extracted headers")]
    system_prompt: Annotated[
        SystemPromptData, Field(description="Extracted system prompt")
    ]
    cached_at: Annotated[
        datetime,
        Field(
            description="Cache timestamp",
            default_factory=lambda: datetime.now(UTC),
        ),
    ] = None  # type: ignore # Pydantic handles this via default_factory

    model_config = ConfigDict(extra="forbid")
