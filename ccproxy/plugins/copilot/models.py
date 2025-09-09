"""Core API models for GitHub Copilot plugin."""

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# Standard OpenAI-compatible models are imported from the centralized location
# to avoid duplication and ensure consistency


# Embedding models - keeping minimal Copilot-specific implementation
class CopilotEmbeddingRequest(BaseModel):
    """Embedding request for Copilot API."""

    input: str | list[str] = Field(..., description="Text to embed")
    model: str = Field(
        default="text-embedding-ada-002", description="Embedding model to use"
    )
    user: str | None = Field(None, description="User identifier")


# Model listing uses standard OpenAI model format


# Error models use the standard OpenAI error format
class CopilotError(BaseModel):
    """Copilot error detail."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(None, description="Parameter that caused error")
    code: str | None = Field(None, description="Error code")


class CopilotErrorResponse(BaseModel):
    """Copilot error response."""

    error: CopilotError = Field(..., description="Error details")


# Utility Models


class CopilotHealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    provider: str = Field(default="copilot", description="Provider name")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Check timestamp"
    )
    details: dict[str, Any] | None = Field(None, description="Additional details")


class CopilotTokenStatus(BaseModel):
    """Token status information."""

    valid: bool = Field(..., description="Whether token is valid")
    expires_at: datetime | None = Field(None, description="Token expiration")
    account_type: str = Field(..., description="Account type")
    copilot_access: bool = Field(..., description="Has Copilot access")
    username: str | None = Field(None, description="GitHub username")


class CopilotQuotaSnapshot(BaseModel):
    """Quota snapshot data for a specific quota type."""

    entitlement: int = Field(..., description="Total quota entitlement")
    overage_count: int = Field(..., description="Number of overages")
    overage_permitted: bool = Field(..., description="Whether overage is allowed")
    percent_remaining: float = Field(..., description="Percentage of quota remaining")
    quota_id: str = Field(..., description="Quota identifier")
    quota_remaining: float = Field(..., description="Remaining quota amount")
    remaining: int = Field(..., description="Remaining quota count")
    unlimited: bool = Field(..., description="Whether quota is unlimited")
    timestamp_utc: str = Field(..., description="Timestamp of last update")


class CopilotUserInternalResponse(BaseModel):
    """User internal response matching upstream /copilot_internal/user endpoint."""

    access_type_sku: str = Field(..., description="Access type SKU")
    analytics_tracking_id: str = Field(..., description="Analytics tracking ID")
    assigned_date: datetime | None = Field(
        None, description="Date when access was assigned"
    )
    can_signup_for_limited: bool = Field(
        ..., description="Can sign up for limited access"
    )
    chat_enabled: bool = Field(..., description="Whether chat is enabled")
    copilot_plan: str = Field(..., description="Copilot plan type")
    organization_login_list: list[str] = Field(
        default_factory=list, description="Organization login list"
    )
    organization_list: list[str] = Field(
        default_factory=list, description="Organization list"
    )
    quota_reset_date: str = Field(..., description="Quota reset date")
    quota_snapshots: dict[str, CopilotQuotaSnapshot] = Field(
        ..., description="Current quota snapshots"
    )
    quota_reset_date_utc: str = Field(..., description="Quota reset date in UTC")


# Authentication Models


class CopilotAuthData(TypedDict, total=False):
    """Authentication data for Copilot/GitHub provider.

    This follows the same pattern as CodexAuthData for consistency.

    Attributes:
        access_token: Bearer token for GitHub Copilot API authentication
        token_type: Token type (typically "bearer")
    """

    access_token: str | None
    token_type: str | None


# Internal Models for Plugin Communication


class CopilotCacheData(BaseModel):
    """Cached detection data for GitHub CLI."""

    cli_available: bool = Field(..., description="Whether GitHub CLI is available")
    cli_version: str | None = Field(None, description="CLI version")
    auth_status: str | None = Field(None, description="Authentication status")
    username: str | None = Field(None, description="Authenticated username")
    last_check: datetime = Field(
        default_factory=datetime.now, description="Last check timestamp"
    )


class CopilotCliInfo(BaseModel):
    """GitHub CLI health information."""

    available: bool = Field(..., description="CLI is available")
    version: str | None = Field(None, description="CLI version")
    authenticated: bool = Field(default=False, description="User is authenticated")
    username: str | None = Field(None, description="Authenticated username")
    error: str | None = Field(None, description="Error message if any")
