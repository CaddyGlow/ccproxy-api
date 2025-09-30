"""Configuration models for the credential balancer plugin."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class RotationStrategy(str, Enum):
    """Supported credential selection strategies."""

    ROUND_ROBIN = "round_robin"
    FAILOVER = "failover"


class CredentialSource(BaseModel):
    """Base model for credential sources."""

    type: Literal["file", "manager"] = Field(
        ..., description="Type of credential source"
    )
    label: str | None = Field(
        default=None,
        description="Optional friendly name used for logging and metrics",
    )

    @property
    def resolved_label(self) -> str:
        """Return a non-empty label for this credential source."""
        return self.label or "unlabeled"


class CredentialFile(CredentialSource):
    """Configuration for a file-based credential source (legacy token snapshots)."""

    type: Literal["file"] = "file"
    path: Path = Field(..., description="Absolute path to a JSON token snapshot file")
    label: str | None = Field(
        default=None,
        description="Optional friendly name used for logging and metrics",
    )

    @field_validator("path", mode="before")
    @classmethod
    def _expand_user_and_env(cls, value: Path | str) -> Path:
        raw_value = str(value)
        expanded = os.path.expandvars(raw_value)
        return Path(expanded).expanduser()

    @field_validator("path")
    @classmethod
    def _ensure_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("credential file paths must be absolute")
        return value

    @model_validator(mode="after")
    def _populate_default_label(self) -> CredentialFile:
        if self.label is None:
            self.label = self.path.stem
        return self

    @property
    def resolved_label(self) -> str:
        """Return a non-empty label for this credential file."""
        return self.label or self.path.stem


class CredentialManager(CredentialSource):
    """Configuration for a manager-based credential source with provider-specific logic."""

    type: Literal["manager"] = "manager"
    manager_key: str = Field(
        ..., description="Auth manager registry key (e.g., 'claude-api', 'codex')"
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Manager-specific configuration options",
    )
    label: str | None = Field(
        default=None,
        description="Optional friendly name used for logging and metrics",
    )

    @model_validator(mode="after")
    def _populate_default_label(self) -> CredentialManager:
        if self.label is None:
            self.label = self.manager_key
        return self

    @property
    def resolved_label(self) -> str:
        """Return a non-empty label for this credential manager."""
        return self.label or self.manager_key


class CredentialPoolConfig(BaseModel):
    """Configuration for an individual credential pool."""

    provider: str = Field(..., description="Internal provider identifier")
    manager_name: str | None = Field(
        default=None,
        description="Registry key to expose this balancer (defaults to '<provider>_credential_balancer')",
    )
    strategy: RotationStrategy = Field(
        default=RotationStrategy.FAILOVER,
        description="How credentials are selected for new requests",
    )
    credentials: list[CredentialFile | CredentialManager] = Field(
        default_factory=list,
        description="Ordered list of credential sources (file or manager) participating in the pool",
    )
    max_failures_before_disable: int = Field(
        default=2,
        ge=1,
        description="Number of failed responses tolerated before disabling a credential",
    )
    cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description="Cooldown window before a failed credential becomes eligible again",
    )
    failure_status_codes: list[int] = Field(
        default_factory=lambda: [401, 403],
        description="HTTP status codes that indicate credential failure",
    )

    @field_validator("credentials")
    @classmethod
    def _ensure_credentials_present(
        cls, value: list[CredentialFile | CredentialManager], _info: ValidationInfo
    ) -> list[CredentialFile | CredentialManager]:
        if not value:
            raise ValueError(
                "credential pool must contain at least one credential file"
            )
        return value

    @field_validator("failure_status_codes")
    @classmethod
    def _validate_status_codes(cls, codes: list[int]) -> list[int]:
        normalised = sorted({code for code in codes if code >= 400})
        if not normalised:
            raise ValueError("at least one failure status code is required")
        return normalised

    @model_validator(mode="after")
    def _apply_default_manager_name(self) -> CredentialPoolConfig:
        if not self.manager_name:
            self.manager_name = f"{self.provider}_credential_balancer"
        return self


class CredentialBalancerSettings(BaseModel):
    """Top-level plugin settings."""

    enabled: bool = Field(default=True, description="Enable credential balancer")
    providers: list[CredentialPoolConfig] = Field(
        default_factory=list, description="Pools managed by the balancer"
    )

    @field_validator("providers")
    @classmethod
    def _ensure_unique_manager_names(
        cls, value: list[CredentialPoolConfig]
    ) -> list[CredentialPoolConfig]:
        seen: set[str] = set()
        for pool in value:
            manager_name = pool.manager_name
            if manager_name is None:
                raise ValueError("manager name resolution failed")
            if manager_name in seen:
                raise ValueError(f"duplicate manager name detected: {manager_name}")
            seen.add(manager_name)
        return value
