"""Reverse proxy configuration settings."""

from typing import Literal

from pydantic import BaseModel, Field


class ReverseProxySettings(BaseModel):
    """Reverse proxy configuration settings."""

    target_url: str = Field(
        default="https://api.anthropic.com",
        description="Target URL for reverse proxy requests",
    )

    timeout: float = Field(
        default=120.0,
        description="Timeout for reverse proxy requests in seconds",
        ge=1.0,
        le=600.0,
    )

    default_mode: Literal["claude_code", "full", "minimal"] = Field(
        default="claude_code",
        description="Default transformation mode for root path reverse proxy, over claude code or auth injection with full",
    )

    claude_code_prefix: str = Field(
        default="/cc",
        description="URL prefix for Claude Code SDK endpoints",
    )
