"""Credential-related models for authentication service."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class CredentialStatus(str, Enum):
    """Status of credentials for validation."""

    VALID = "valid"
    MISSING = "missing"
    EXPIRED = "expired"
    INVALID = "invalid"


class CredentialValidation(BaseModel):
    """Result of credential validation."""

    status: CredentialStatus
    message: str | None = None
    expires_at: datetime | None = None
