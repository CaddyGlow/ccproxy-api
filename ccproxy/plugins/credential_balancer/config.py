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

    type: Literal["manager"] = Field(
        default="manager", description="Type of credential source"
    )
    label: str | None = Field(
        default=None,
        description="Optional friendly name used for logging and metrics",
    )

    @property
    def resolved_label(self) -> str:
        """Return a non-empty label for this credential source."""
        return self.label or "unlabeled"


class CredentialManager(CredentialSource):
    """Configuration for a manager-based credential source with provider-specific logic.

    Specify either manager_key (registry lookup) or manager_class (direct import).

    The config dict supports additional options:

    **Storage options:**
    - `enable_backups` (bool): Create timestamped backups before overwriting credentials (default: True)

    **Manager options:**
    - `credentials_ttl` (float): Seconds to cache credentials before rechecking storage (default: 30.0)
    - `refresh_grace_seconds` (float): Seconds before expiry to trigger proactive token refresh (default: 120.0)

    Example:
        ```toml
        { type = "manager",
          file = "~/.config/ccproxy/codex_pro.json",
          config = {
            enable_backups = true,
            credentials_ttl = 60.0,
            refresh_grace_seconds = 300.0
          }
        }
        ```
    """

    type: Literal["manager"] = "manager"
    file: Path | None = Field(
        default=None,
        description="Path to custom credential file (overrides default storage location)",
    )
    manager_key: str | None = Field(
        default=None,
        description="Auth manager registry key (e.g., 'codex', 'claude-api'). Mutually exclusive with manager_class.",
    )
    manager_class: str | None = Field(
        default=None,
        description="Fully qualified manager class name (e.g., 'ccproxy.plugins.oauth_codex.manager.CodexTokenManager'). Mutually exclusive with manager_key.",
    )
    storage_class: str | None = Field(
        default=None,
        description="Fully qualified storage class name (e.g., 'ccproxy.plugins.oauth_codex.storage.CodexTokenStorage'). Required when using manager_class with custom file.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional manager and storage configuration options (see class docstring for supported keys)",
    )
    label: str | None = Field(
        default=None,
        description="Optional friendly name used for logging and metrics",
    )

    @field_validator("file", mode="before")
    @classmethod
    def _expand_file_path(cls, value: Path | str | None) -> Path | None:
        """Expand environment variables and user home directory in file path."""
        if value is None:
            return None
        raw_value = str(value)
        expanded = os.path.expandvars(raw_value)
        return Path(expanded).expanduser()

    @model_validator(mode="after")
    def _validate_manager_specification(self) -> CredentialManager:
        # Allow both to be None - they may be inherited from pool-level defaults
        # But if both are specified, that's an error
        if self.manager_key and self.manager_class:
            raise ValueError(
                "manager_key and manager_class are mutually exclusive, specify only one"
            )
        # If using manager_class with custom file, storage_class is required
        # (unless it will be inherited from pool defaults)
        if self.manager_class and self.file and not self.storage_class:
            raise ValueError(
                "storage_class is required when using manager_class with custom file path"
            )
        return self

    @model_validator(mode="after")
    def _populate_default_label(self) -> CredentialManager:
        if self.label is None:
            if self.manager_key:
                self.label = self.manager_key
            elif self.manager_class:
                # Extract class name from fully qualified path
                self.label = self.manager_class.rsplit(".", 1)[-1]
            else:
                self.label = "unlabeled"
        return self

    @property
    def resolved_label(self) -> str:
        """Return a non-empty label for this credential manager."""
        if self.label:
            return self.label
        if self.manager_key:
            return self.manager_key
        if self.manager_class:
            return self.manager_class.rsplit(".", 1)[-1]
        return "unlabeled"


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
    manager_class: str | None = Field(
        default=None,
        description="Default manager class for all credentials in this pool (can be overridden per credential)",
    )
    storage_class: str | None = Field(
        default=None,
        description="Default storage class for all credentials in this pool (can be overridden per credential)",
    )
    credentials: list[CredentialManager] = Field(
        default_factory=list,
        description="Ordered list of manager-based credential sources participating in the pool",
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
        cls, value: list[CredentialManager], _info: ValidationInfo
    ) -> list[CredentialManager]:
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

    @model_validator(mode="after")
    def _apply_pool_defaults_to_credentials(self) -> CredentialPoolConfig:
        """Apply pool-level manager_class and storage_class to credentials that don't specify them."""
        if not self.manager_class and not self.storage_class:
            # No pool-level defaults to apply
            return self

        for cred in self.credentials:
            # Only apply to CredentialManager type
            if isinstance(cred, CredentialManager):
                # Apply pool-level manager_class if credential doesn't specify one
                if (
                    self.manager_class
                    and not cred.manager_class
                    and not cred.manager_key
                ):
                    cred.manager_class = self.manager_class

                # Apply pool-level storage_class if credential doesn't specify one
                if self.storage_class and not cred.storage_class:
                    cred.storage_class = self.storage_class

        return self

    @model_validator(mode="after")
    def _validate_credentials_after_defaults(self) -> CredentialPoolConfig:
        """Validate that all credentials have required manager information after applying defaults."""
        for idx, cred in enumerate(self.credentials):
            if isinstance(cred, CredentialManager):
                # After applying defaults, each credential must have either manager_key or manager_class
                if not cred.manager_key and not cred.manager_class:
                    raise ValueError(
                        f"Credential at index {idx} missing manager specification. "
                        f"Either set manager_key/manager_class on the credential, "
                        f"or set manager_class at pool level."
                    )
                # If using manager_class with file, storage_class is required
                if cred.manager_class and cred.file and not cred.storage_class:
                    raise ValueError(
                        f"Credential at index {idx} with manager_class and file path "
                        f"requires storage_class (either on credential or at pool level)"
                    )
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
