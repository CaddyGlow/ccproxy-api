"""Pydantic models for metrics data in Claude Code Proxy API Server."""

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class UserAgentCategory(str, Enum):
    """User agent categories for metrics tracking."""

    PYTHON_SDK = "python_sdk"
    NODEJS = "nodejs"
    CURL = "curl"
    POSTMAN = "postman"
    BROWSER = "browser"
    ANTHROPIC_SDK = "anthropic_sdk"
    OPENAI_SDK = "openai_sdk"
    OTHER = "other"


class HTTPMetrics(BaseModel):
    """HTTP request metrics data."""

    method: Annotated[str, Field(description="HTTP method (GET, POST, etc.)")]
    endpoint: Annotated[str, Field(description="API endpoint path")]
    status_code: Annotated[int, Field(description="HTTP status code")]
    api_type: Annotated[str, Field(description="API type (anthropic, openai)")]
    user_agent_category: Annotated[
        UserAgentCategory, Field(description="Categorized user agent")
    ]
    duration_seconds: Annotated[
        float, Field(description="Request duration in seconds", ge=0)
    ]
    request_size_bytes: Annotated[
        int, Field(description="Request size in bytes", ge=0)
    ] = 0
    response_size_bytes: Annotated[
        int, Field(description="Response size in bytes", ge=0)
    ] = 0
    user_agent: Annotated[str, Field(description="Raw user agent string")] = ""
    error_type: Annotated[
        str | None, Field(description="Error type if request failed")
    ] = None

    # Rate limit fields
    rate_limit_requests_limit: Annotated[
        int | None, Field(description="API key rate limit for requests")
    ] = None
    rate_limit_requests_remaining: Annotated[
        int | None, Field(description="API key rate limit remaining requests")
    ] = None
    rate_limit_tokens_limit: Annotated[
        int | None, Field(description="API key rate limit for tokens")
    ] = None
    rate_limit_tokens_remaining: Annotated[
        int | None, Field(description="API key rate limit remaining tokens")
    ] = None
    rate_limit_reset_timestamp: Annotated[
        str | None, Field(description="API key rate limit reset timestamp")
    ] = None
    retry_after_seconds: Annotated[
        int | None, Field(description="Retry-after header value in seconds")
    ] = None

    # OAuth unified rate limit fields
    oauth_unified_status: Annotated[
        str | None, Field(description="OAuth unified rate limit status")
    ] = None
    oauth_unified_claim: Annotated[
        str | None, Field(description="OAuth unified rate limit claim")
    ] = None
    oauth_unified_fallback_percentage: Annotated[
        float | None, Field(description="OAuth unified fallback percentage")
    ] = None
    oauth_unified_reset: Annotated[
        str | None, Field(description="OAuth unified rate limit reset time")
    ] = None

    # Authentication type
    auth_type: Annotated[
        str | None, Field(description="Authentication type (api_key/oauth)")
    ] = None

    model_config = ConfigDict(extra="forbid")


class ModelMetrics(BaseModel):
    """Model-specific metrics data."""

    model: Annotated[
        str, Field(description="Model name (e.g., claude-3-5-sonnet-20241022)")
    ]
    api_type: Annotated[str, Field(description="API type (anthropic, openai)")]
    endpoint: Annotated[str, Field(description="API endpoint path")]
    streaming: Annotated[bool, Field(description="Whether request was streaming")]
    input_tokens: Annotated[
        int, Field(description="Number of input tokens processed", ge=0)
    ] = 0
    output_tokens: Annotated[
        int, Field(description="Number of output tokens generated", ge=0)
    ] = 0
    cache_creation_input_tokens: Annotated[
        int, Field(description="Number of tokens used for cache creation", ge=0)
    ] = 0
    cache_read_input_tokens: Annotated[
        int, Field(description="Number of tokens read from cache", ge=0)
    ] = 0
    estimated_cost: Annotated[
        float, Field(description="Estimated cost in USD", ge=0)
    ] = 0.0

    model_config = ConfigDict(extra="forbid")


class ErrorMetrics(BaseModel):
    """Error metrics data."""

    error_type: Annotated[str, Field(description="Error type (e.g., rate_limit, auth)")]
    endpoint: Annotated[str, Field(description="API endpoint path")]
    status_code: Annotated[int, Field(description="HTTP status code")]
    api_type: Annotated[str, Field(description="API type (anthropic, openai)")]
    user_agent_category: Annotated[
        UserAgentCategory, Field(description="Categorized user agent")
    ]

    model_config = ConfigDict(extra="forbid")


class MetricsSnapshot(BaseModel):
    """Complete metrics snapshot for export."""

    http_metrics: Annotated[
        list[HTTPMetrics], Field(description="HTTP request metrics")
    ] = []
    model_metrics: Annotated[
        list[ModelMetrics], Field(description="Model usage metrics")
    ] = []
    error_metrics: Annotated[
        list[ErrorMetrics], Field(description="Error metrics")
    ] = []
    active_requests: Annotated[
        int, Field(description="Current number of active requests", ge=0)
    ] = 0

    model_config = ConfigDict(extra="forbid")
