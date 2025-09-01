"""Access log schema and payload definitions for observability logging.

This plugin-local module owns the SQLModel table schema and the associated
payload TypedDict used by the DuckDB storage and analytics layers.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from sqlmodel import Field, SQLModel


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
    num_turns: int = Field(default=0)  # number of conversation turns

    # Session context metadata
    session_type: str = Field(default="")  # "session_pool" or "direct"
    session_status: str = Field(default="")  # active, idle, connecting, etc.
    session_age_seconds: float = Field(default=0.0)  # how long session has been alive
    session_message_count: int = Field(default=0)  # number of messages in session
    session_client_id: str = Field(default="")  # unique session client identifier
    session_pool_enabled: bool = Field(
        default=False
    )  # whether session pooling is enabled
    session_idle_seconds: float = Field(default=0.0)  # how long since last activity
    session_error_count: int = Field(default=0)  # number of errors in this session
    session_is_new: bool = Field(
        default=True
    )  # whether this is a newly created session

    class Config:
        """SQLModel configuration."""

        # Enable automatic conversion from dict
        from_attributes = True
        # Use enum values
        use_enum_values = True


class AccessLogPayload(TypedDict, total=False):
    """TypedDict for access log data payloads."""

    # Core request identification
    request_id: str
    timestamp: int | float | datetime

    # Request details
    method: str
    endpoint: str
    path: str
    query: str
    client_ip: str
    user_agent: str

    # Service and model info
    service_type: str
    model: str
    streaming: bool

    # Response details
    status_code: int
    duration_ms: float
    duration_seconds: float

    # Token and cost tracking
    tokens_input: int
    tokens_output: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    cost_sdk_usd: float
    num_turns: int

    # Session context metadata
    session_type: str
    session_status: str
    session_age_seconds: float
    session_message_count: int
    session_client_id: str
    session_pool_enabled: bool
    session_idle_seconds: float
    session_error_count: int
    session_is_new: bool

