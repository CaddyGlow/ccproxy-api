"""Pydantic models for confirmation system."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConfirmationStatus(Enum):
    """Status of a confirmation request."""

    PENDING = "pending"
    ALLOWED = "allowed"
    DENIED = "denied"
    EXPIRED = "expired"


class ConfirmationInput(BaseModel):
    """Input parameters for a tool confirmation request."""

    command: str | None = None
    code: str | None = None
    path: str | None = None
    content: str | None = None
    # Add other common input fields as needed


class ConfirmationRequest(BaseModel):
    """Represents a permission confirmation request."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    input: dict[str, str]  # More specific than Any
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    resolved_at: datetime | None = None

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.status != ConfirmationStatus.PENDING:
            return False
        return datetime.utcnow() > self.expires_at

    def time_remaining(self) -> int:
        """Get time remaining in seconds."""
        if self.status != ConfirmationStatus.PENDING:
            return 0
        remaining = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))

    def resolve(self, allowed: bool) -> None:
        """Resolve the request."""
        if self.status != ConfirmationStatus.PENDING:
            raise ValueError(f"Cannot resolve request in {self.status} status")

        self.status = (
            ConfirmationStatus.ALLOWED if allowed else ConfirmationStatus.DENIED
        )
        self.resolved_at = datetime.utcnow()


class ConfirmationEvent(BaseModel):
    """Event emitted by the confirmation service."""

    type: str  # "confirmation_request", "confirmation_resolved", "confirmation_expired"
    request_id: str
    tool_name: str | None = None
    input: dict[str, str] | None = None
    created_at: str | None = None
    expires_at: str | None = None
    timeout_seconds: int | None = None
    allowed: bool | None = None
    resolved_at: str | None = None
    expired_at: str | None = None
    message: str | None = None
