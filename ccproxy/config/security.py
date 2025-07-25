"""Security configuration settings."""

from pydantic import BaseModel, Field


class SecuritySettings(BaseModel):
    """Security-specific configuration settings."""

    auth_token: str | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )

    confirmation_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout in seconds for permission confirmation requests (5-300)",
    )
