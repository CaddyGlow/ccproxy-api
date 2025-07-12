"""Security configuration settings."""

from typing import Literal

from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    """Security-specific configuration settings."""

    auth_token: str | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )
