"""Observability configuration settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ObservabilitySettings(BaseModel):
    """Observability configuration settings."""

    # Endpoint Controls
    metrics_endpoint_enabled: bool = Field(
        default=False,
        description="Enable Prometheus /metrics endpoint",
    )

    logs_endpoints_enabled: bool = Field(
        default=False,
        description="Enable logs query/analytics/streaming endpoints (/logs/*)",
    )

    dashboard_enabled: bool = Field(
        default=False,
        description="Enable metrics dashboard endpoint (/dashboard)",
    )

    # Data Collection & Storage
    logs_collection_enabled: bool = Field(
        default=False,
        description="Enable collection of request/response logs to storage backend",
    )

    log_storage_backend: Literal["duckdb", "none"] = Field(
        default="duckdb",
        description="Storage backend for logs ('duckdb' or 'none')",
    )

    # Storage Configuration
    duckdb_path: str = Field(
        default_factory=lambda: str(
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / "ccproxy"
            / "metrics.duckdb"
        ),
        description="Path to DuckDB database file",
    )

    # Pushgateway configuration removed - functionality moved to metrics plugin
    # The metrics plugin now manages its own pushgateway settings

    # Stats printing configuration
    stats_printing_format: str = Field(
        default="console",
        description="Format for stats output: 'console', 'rich', 'log', 'json'",
    )

    @model_validator(mode="after")
    def check_feature_dependencies(self) -> ObservabilitySettings:
        """Validate feature dependencies to prevent invalid configurations."""
        # Dashboard requires logs endpoints (functional dependency)
        if self.dashboard_enabled and not self.logs_endpoints_enabled:
            raise ValueError(
                "Cannot enable 'dashboard_enabled' without 'logs_endpoints_enabled'. "
                "Dashboard needs logs API to function."
            )

        # Logs endpoints require storage to query from
        if self.logs_endpoints_enabled and self.log_storage_backend == "none":
            raise ValueError(
                "Cannot enable 'logs_endpoints_enabled' when 'log_storage_backend' is 'none'. "
                "Logs endpoints need storage backend to query data."
            )

        # Log collection requires storage to write to
        if self.logs_collection_enabled and self.log_storage_backend == "none":
            raise ValueError(
                "Cannot enable 'logs_collection_enabled' when 'log_storage_backend' is 'none'. "
                "Collection needs storage backend to persist data."
            )

        return self

    @field_validator("stats_printing_format")
    @classmethod
    def validate_stats_printing_format(cls, v: str) -> str:
        """Validate and normalize stats printing format."""
        lower_v = v.lower()
        valid_formats = ["console", "rich", "log", "json"]
        if lower_v not in valid_formats:
            raise ValueError(
                f"Invalid stats printing format: {v}. Must be one of {valid_formats}"
            )
        return lower_v

    @property
    def needs_storage_backend(self) -> bool:
        """Check if any feature requires storage backend initialization."""
        return self.logs_endpoints_enabled or self.logs_collection_enabled

    @property
    def any_endpoint_enabled(self) -> bool:
        """Check if any observability endpoint is enabled."""
        return (
            self.metrics_endpoint_enabled
            or self.logs_endpoints_enabled
            or self.dashboard_enabled
        )

    # Backward compatibility properties
    @property
    def metrics_enabled(self) -> bool:
        """Backward compatibility: True if any metrics feature is enabled."""
        return self.any_endpoint_enabled

    @property
    def duckdb_enabled(self) -> bool:
        """Backward compatibility: True if DuckDB storage backend is selected."""
        return self.log_storage_backend == "duckdb"

    @property
    def enabled(self) -> bool:
        """Check if observability is enabled."""
        return self.any_endpoint_enabled
