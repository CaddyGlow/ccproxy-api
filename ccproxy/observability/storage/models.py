"""
SQLModel schema definitions for observability storage.

This module provides the centralized schema definitions for access logs and metrics
using SQLModel to ensure type safety and eliminate column name repetition.
"""

from datetime import datetime
from typing import Optional

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

    class Config:
        """SQLModel configuration."""

        # Enable automatic conversion from dict
        from_attributes = True
        # Use enum values
        use_enum_values = True
