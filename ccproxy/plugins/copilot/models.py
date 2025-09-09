"""Core API models for GitHub Copilot plugin."""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# Request Models


class CopilotMessage(BaseModel):
    """Message in a chat completion request."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(..., description="Content of the message")
    name: str | None = Field(None, description="Name of the sender (optional)")


class CopilotChatRequest(BaseModel):
    """Chat completion request for Copilot API."""

    messages: list[CopilotMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    model: str = Field(default="gpt-4", description="Model to use for completion")
    temperature: float | None = Field(
        None, ge=0, le=2, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        None, ge=1, description="Maximum number of tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(
        None, ge=-2, le=2, description="Presence penalty"
    )
    frequency_penalty: float | None = Field(
        None, ge=-2, le=2, description="Frequency penalty"
    )
    top_p: float | None = Field(
        None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    n: int | None = Field(None, ge=1, description="Number of completions to generate")
    user: str | None = Field(None, description="User identifier for abuse monitoring")


class CopilotEmbeddingRequest(BaseModel):
    """Embedding request for Copilot API."""

    input: str | list[str] = Field(..., description="Text to embed")
    model: str = Field(
        default="text-embedding-ada-002", description="Embedding model to use"
    )
    user: str | None = Field(None, description="User identifier")


# Response Models


class CopilotUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., description="Tokens in the prompt")
    completion_tokens: int | None = Field(None, description="Tokens in the completion")
    total_tokens: int = Field(..., description="Total tokens used")


class CopilotChoice(BaseModel):
    """A completion choice."""

    index: int = Field(..., description="Choice index")
    message: CopilotMessage = Field(..., description="Generated message")
    finish_reason: str | None = Field(None, description="Reason for finishing")


class CopilotChatResponse(BaseModel):
    """Chat completion response from Copilot API."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid4().hex}", description="Response ID"
    )
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Creation timestamp",
    )
    model: str = Field(..., description="Model used")
    choices: list[CopilotChoice] = Field(..., description="Generated choices")
    usage: CopilotUsage | None = Field(None, description="Token usage")


class CopilotStreamChoice(BaseModel):
    """A streaming completion choice."""

    index: int = Field(..., description="Choice index")
    delta: dict[str, Any] = Field(..., description="Delta content")
    finish_reason: str | None = Field(None, description="Reason for finishing")


class CopilotStreamResponse(BaseModel):
    """Streaming chat completion response chunk."""

    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[CopilotStreamChoice] = Field(..., description="Streaming choices")


class CopilotEmbeddingData(BaseModel):
    """Embedding data for a single input."""

    object: str = Field(default="embedding", description="Object type")
    embedding: list[float] = Field(..., description="Embedding vector")
    index: int = Field(..., description="Input index")


class CopilotEmbeddingResponse(BaseModel):
    """Embedding response from Copilot API."""

    object: str = Field(default="list", description="Object type")
    data: list[CopilotEmbeddingData] = Field(..., description="Embedding data")
    model: str = Field(..., description="Model used")
    usage: CopilotUsage = Field(..., description="Token usage")


class CopilotModel(BaseModel):
    """Available model information."""

    id: str = Field(..., description="Model ID")
    object: str = Field(default="model", description="Object type")
    created: int = Field(
        default_factory=lambda: int(datetime.now().timestamp()),
        description="Creation timestamp",
    )
    owned_by: str = Field(default="github", description="Model owner")


class CopilotModelsResponse(BaseModel):
    """Response containing available models."""

    object: str = Field(default="list", description="Object type")
    data: list[CopilotModel] = Field(..., description="Available models")


# Error Models


class CopilotError(BaseModel):
    """Error response from Copilot API."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    param: str | None = Field(None, description="Parameter that caused error")
    code: str | None = Field(None, description="Error code")


class CopilotErrorResponse(BaseModel):
    """Error response wrapper."""

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
