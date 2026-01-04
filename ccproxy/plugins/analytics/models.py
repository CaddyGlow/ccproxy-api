"""Access log schema and payload definitions (owned by analytics)."""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel
from typing_extensions import TypedDict


class AccessLog(SQLModel, table=True):
    """Access log model for storing request/response data."""

    __tablename__ = "access_logs"

    # Core request identification
    request_id: str = Field(primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)

    # Request details
    method: str
    endpoint: str
    path: str
    query: str = Field(default="")
    client_ip: str
    user_agent: str

    # Service and model info
    service_type: str
    provider: str = Field(default="")
    model: str
    streaming: bool = Field(default=False)

    # Response details
    status_code: int
    duration_ms: float
    duration_seconds: float

    # Token and cost tracking
    tokens_input: int = Field(default=0)
    tokens_output: int = Field(default=0)
    cache_read_tokens: int = Field(default=0)
    cache_write_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    cost_sdk_usd: float = Field(default=0.0)
    num_turns: int = Field(default=0)

    # Session context metadata
    session_type: str = Field(default="")
    session_status: str = Field(default="")
    session_age_seconds: float = Field(default=0.0)
    session_message_count: int = Field(default=0)
    session_client_id: str = Field(default="")
    session_pool_enabled: bool = Field(default=False)
    session_idle_seconds: float = Field(default=0.0)
    session_error_count: int = Field(default=0)
    session_is_new: bool = Field(default=True)

    # SQLModel provides its own config typing; avoid overriding with Pydantic ConfigDict
    # from_attributes=True is not required for SQLModel usage here
    # Keep default SQLModel config to satisfy mypy type expectations


class AccessLogPayload(TypedDict, total=False):
    """TypedDict for access log data payloads."""

    request_id: str
    timestamp: int | float | datetime
    method: str
    endpoint: str
    path: str
    query: str
    client_ip: str
    user_agent: str
    service_type: str
    provider: str
    model: str
    streaming: bool
    status_code: int
    duration_ms: float
    duration_seconds: float
    tokens_input: int
    tokens_output: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    cost_sdk_usd: float
    num_turns: int
    session_type: str
    session_status: str
    session_age_seconds: float
    session_message_count: int
    session_client_id: str
    session_pool_enabled: bool
    session_idle_seconds: float
    session_error_count: int
    session_is_new: bool
