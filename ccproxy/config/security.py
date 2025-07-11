"""Security configuration settings."""

from typing import Literal

from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    """Security-specific configuration settings."""

    auth_token: str | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )

    api_tools_handling: Literal["error", "warning", "ignore"] = Field(
        default="warning",
        description="How to handle tools definitions in requests: error, warning, or ignore",
    )
