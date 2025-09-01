"""Authentication-related settings."""

from pydantic import BaseModel, ConfigDict, Field


class AuthSettings(BaseModel):
    """Configuration for authentication behavior and caching."""

    model_config = ConfigDict(extra="ignore")

    credentials_ttl_seconds: float = Field(
        3600.0,
        description=(
            "Cache duration for loaded credentials before rechecking storage. "
            "Use nested env var AUTH__CREDENTIALS_TTL_SECONDS to override."
        ),
        ge=0.0,
    )


__all__ = ["AuthSettings"]
