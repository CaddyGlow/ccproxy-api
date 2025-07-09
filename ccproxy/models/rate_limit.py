"""Rate limit models for Claude Proxy API Server."""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StandardRateLimit(BaseModel):
    """Standard rate limit data for API key authentication."""

    requests_limit: Annotated[
        int | None,
        Field(description="Maximum number of requests allowed in the time window"),
    ] = None
    requests_remaining: Annotated[
        int | None,
        Field(description="Number of requests remaining in the current time window"),
    ] = None
    tokens_limit: Annotated[
        int | None,
        Field(description="Maximum number of tokens allowed in the time window"),
    ] = None
    tokens_remaining: Annotated[
        int | None,
        Field(description="Number of tokens remaining in the current time window"),
    ] = None
    reset_timestamp: Annotated[
        datetime | None, Field(description="When the rate limit window resets")
    ] = None
    retry_after_seconds: Annotated[
        int | None,
        Field(description="Seconds to wait before retrying after rate limit"),
    ] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("retry_after_seconds")
    @classmethod
    def validate_retry_after_seconds(cls, v: int | None) -> int | None:
        """Validate retry_after_seconds is non-negative."""
        if v is not None and v < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        return v


class OAuthUnifiedRateLimit(BaseModel):
    """OAuth unified rate limit data for OAuth authentication."""

    status: Annotated[
        str | None, Field(description="Rate limit status (allowed/denied)")
    ] = None
    representative_claim: Annotated[
        str | None, Field(description="Representative claim (e.g., five_hour)")
    ] = None
    fallback_percentage: Annotated[
        float | None, Field(description="Fallback percentage for rate limiting")
    ] = None
    reset_timestamp: Annotated[
        datetime | None, Field(description="When the rate limit window resets")
    ] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("fallback_percentage")
    @classmethod
    def validate_fallback_percentage(cls, v: float | None) -> float | None:
        """Validate fallback_percentage is between 0 and 100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("fallback_percentage must be between 0 and 100")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        """Validate status is one of the allowed values."""
        if v is not None and v not in ["allowed", "denied"]:
            raise ValueError("status must be 'allowed' or 'denied'")
        return v


class RateLimitData(BaseModel):
    """Container model for rate limit data."""

    auth_type: Annotated[
        Literal["api_key", "oauth", "unknown"],
        Field(description="Type of authentication used"),
    ]
    standard: Annotated[
        StandardRateLimit | None, Field(description="Standard API key rate limit data")
    ] = None
    oauth_unified: Annotated[
        OAuthUnifiedRateLimit | None, Field(description="OAuth unified rate limit data")
    ] = None
    timestamp: Annotated[
        datetime, Field(description="When the rate limit data was collected")
    ]

    model_config = ConfigDict(extra="forbid")

    @field_validator("standard", "oauth_unified")
    @classmethod
    def validate_rate_limit_data_consistency(
        cls, v: StandardRateLimit | OAuthUnifiedRateLimit | None, info: Any
    ) -> StandardRateLimit | OAuthUnifiedRateLimit | None:
        """Validate that appropriate rate limit data is present for auth type."""
        if info.context and "auth_type" in info.context:
            auth_type = info.context["auth_type"]
            field_name = info.field_name

            if auth_type == "api_key" and field_name == "standard" and v is None:
                raise ValueError(
                    "standard rate limit data required for api_key auth_type"
                )
            elif auth_type == "oauth" and field_name == "oauth_unified" and v is None:
                raise ValueError(
                    "oauth_unified rate limit data required for oauth auth_type"
                )

        return v


class RateLimitStatus(BaseModel):
    """Current rate limit status tracking."""

    auth_type: Annotated[str, Field(description="Type of authentication used")]
    is_limited: Annotated[
        bool, Field(description="Whether the client is currently rate limited")
    ]
    utilization_percentage: Annotated[
        float | None, Field(description="Current utilization percentage (0-100)")
    ] = None
    time_until_reset: Annotated[
        int | None, Field(description="Seconds until rate limit resets")
    ] = None
    estimated_exhaustion_time: Annotated[
        datetime | None,
        Field(description="Estimated time when rate limit will be exhausted"),
    ] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("utilization_percentage")
    @classmethod
    def validate_utilization_percentage(cls, v: float | None) -> float | None:
        """Validate utilization_percentage is between 0 and 100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("utilization_percentage must be between 0 and 100")
        return v

    @field_validator("time_until_reset")
    @classmethod
    def validate_time_until_reset(cls, v: int | None) -> int | None:
        """Validate time_until_reset is non-negative."""
        if v is not None and v < 0:
            raise ValueError("time_until_reset must be non-negative")
        return v


__all__ = [
    "StandardRateLimit",
    "OAuthUnifiedRateLimit",
    "RateLimitData",
    "RateLimitStatus",
]
