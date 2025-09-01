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

    # Data Collection
    logs_collection_enabled: bool = Field(
        default=False,
        description="Enable collection of request/response logs to storage backend",
    )
    # Storage configuration moved to duckdb_storage plugin

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

        # Storage dependency checks are handled by the storage plugin.

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
        """Backward compatibility: retained for callers; always True when endpoints/collection need storage.

        Actual backend selection is now handled by the duckdb_storage plugin.
        """
        return self.needs_storage_backend

    @property
    def enabled(self) -> bool:
        """Check if observability is enabled."""
        return self.any_endpoint_enabled
