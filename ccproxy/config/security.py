"""Security configuration settings."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator


class SecuritySettings(BaseModel):
    """Security-specific configuration settings."""

    auth_token: SecretStr | None = Field(
        default=None,
        description="Bearer token for API authentication (optional)",
    )

    @field_validator("auth_token", mode="before")
    @classmethod
    def validate_auth_token(cls, v: Any) -> Any:
        """Convert string values to SecretStr."""
        if v is None:
            return None
        if isinstance(v, str):
            return SecretStr(v)
        return v

    confirmation_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout in seconds for permission confirmation requests (5-300)",
    )
