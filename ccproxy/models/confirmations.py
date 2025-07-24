"""Pydantic models for confirmation system."""

import asyncio
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import NotRequired, TypedDict

from pydantic import BaseModel, Field, PrivateAttr


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

    # Private attribute for event-driven waiting
    _resolved_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.status != ConfirmationStatus.PENDING:
            return False

        # Handle both timezone-aware and naive datetimes
        now = datetime.utcnow()
        expires_at = self.expires_at

        # If expires_at is timezone-aware, convert now to timezone-aware
        if expires_at.tzinfo is not None:
            from datetime import timezone

            now = now.replace(tzinfo=UTC)
        # If expires_at is naive but now is timezone-aware, make expires_at timezone-aware
        elif hasattr(now, "tzinfo") and now.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=UTC)

        return now > expires_at

    def time_remaining(self) -> int:
        """Get time remaining in seconds."""
        if self.status != ConfirmationStatus.PENDING:
            return 0

        # Handle both timezone-aware and naive datetimes
        now = datetime.utcnow()
        expires_at = self.expires_at

        # If expires_at is timezone-aware, convert now to timezone-aware
        if expires_at.tzinfo is not None:
            from datetime import timezone

            now = now.replace(tzinfo=UTC)
        # If expires_at is naive but now is timezone-aware, make expires_at timezone-aware
        elif hasattr(now, "tzinfo") and now.tzinfo is not None:
            expires_at = expires_at.replace(tzinfo=UTC)

        remaining = (expires_at - now).total_seconds()
        return max(0, int(remaining))

    def resolve(self, allowed: bool) -> None:
        """Resolve the request."""
        if self.status != ConfirmationStatus.PENDING:
            raise ValueError(f"Cannot resolve request in {self.status} status")

        self.status = (
            ConfirmationStatus.ALLOWED if allowed else ConfirmationStatus.DENIED
        )
        self.resolved_at = datetime.utcnow()
        # Signal waiting coroutines that resolution is complete
        self._resolved_event.set()


class ConfirmationEventDict(TypedDict):
    """Typed dictionary for confirmation event data."""

    type: str
    request_id: str
    tool_name: NotRequired[str]
    input: NotRequired[dict[str, str]]
    created_at: NotRequired[str]
    expires_at: NotRequired[str]
    timeout_seconds: NotRequired[int]
    allowed: NotRequired[bool]
    resolved_at: NotRequired[str]
    expired_at: NotRequired[str]
    message: NotRequired[str]


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
