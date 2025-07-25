"""Pydantic models for permission system."""

import asyncio
import uuid
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr


class PermissionStatus(Enum):
    """Status of a permission request."""

    PENDING = "pending"
    ALLOWED = "allowed"
    DENIED = "denied"
    EXPIRED = "expired"


class EventType(Enum):
    """Types of permission events."""

    PERMISSION_REQUEST = "permission_request"
    PERMISSION_RESOLVED = "permission_resolved"
    PERMISSION_EXPIRED = "permission_expired"


class PermissionInput(BaseModel):
    """Input parameters for a tool permission request."""

    command: str | None = None
    code: str | None = None
    path: str | None = None
    content: str | None = None
    # Add other common input fields as needed


class PermissionRequest(BaseModel):
    """Represents a tool permission request."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    input: dict[str, str]  # More specific than Any
    status: PermissionStatus = PermissionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime
    resolved_at: datetime | None = None

    # Private attribute for event-driven waiting
    _resolved_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        if self.status != PermissionStatus.PENDING:
            return False

        now, expires_at = self._normalize_datetimes(datetime.now(UTC), self.expires_at)
        return now > expires_at

    def time_remaining(self) -> int:
        """Get time remaining in seconds."""
        if self.status != PermissionStatus.PENDING:
            return 0

        now, expires_at = self._normalize_datetimes(datetime.now(UTC), self.expires_at)
        remaining = (expires_at - now).total_seconds()
        return max(0, int(remaining))

    def resolve(self, allowed: bool) -> None:
        """Resolve the request."""
        if self.status != PermissionStatus.PENDING:
            raise ValueError(f"Cannot resolve request in {self.status} status")

        self.status = PermissionStatus.ALLOWED if allowed else PermissionStatus.DENIED
        self.resolved_at = datetime.now(UTC)
        # Signal waiting coroutines that resolution is complete
        self._resolved_event.set()

    def _normalize_datetimes(
        self, dt1: datetime, dt2: datetime
    ) -> tuple[datetime, datetime]:
        """Normalize two datetimes to ensure both are timezone-aware.

        Args:
            dt1: First datetime to normalize
            dt2: Second datetime to normalize

        Returns:
            Tuple of normalized timezone-aware datetimes
        """
        # If dt1 is timezone-aware, convert dt2 to timezone-aware if needed
        if dt1.tzinfo is not None:
            if dt2.tzinfo is None:
                dt2 = dt2.replace(tzinfo=UTC)
        # If dt2 is timezone-aware, convert dt1 to timezone-aware if needed
        elif dt2.tzinfo is not None:
            dt1 = dt1.replace(tzinfo=UTC)

        return dt1, dt2


class PermissionEvent(BaseModel):
    """Event emitted by the permission service."""

    type: EventType
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
