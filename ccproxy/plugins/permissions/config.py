"""Configuration for permissions plugin."""

from pydantic import BaseModel, Field


class PermissionsConfig(BaseModel):
    """Configuration for the permissions plugin."""

    enabled: bool = Field(
        default=True,
        description="Enable the permissions service",
    )
    timeout_seconds: int = Field(
        default=30,
        description="Default timeout for permission requests in seconds",
    )
    enable_terminal_ui: bool = Field(
        default=True,
        description="Enable terminal UI for permission requests",
    )
    enable_sse_stream: bool = Field(
        default=True,
        description="Enable SSE streaming endpoint for external handlers",
    )
    cleanup_after_minutes: int = Field(
        default=5,
        description="Minutes to keep resolved requests before cleanup",
    )
