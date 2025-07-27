"""Scheduler configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerSettings(BaseSettings):
    """
    Configuration settings for the unified scheduler system.

    Controls global scheduler behavior and individual task configurations.
    Settings can be configured via environment variables with SCHEDULER__ prefix.
    """

    # Global scheduler settings
    enabled: bool = Field(
        default=True,
        description="Whether the scheduler system is enabled",
    )

    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of tasks that can run concurrently",
    )

    graceful_shutdown_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout in seconds for graceful task shutdown",
    )

    # Pricing updater task settings
    pricing_update_enabled: bool = Field(
        default=True,
        description="Whether pricing cache update task is enabled",
    )

    pricing_update_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Interval in hours between pricing cache updates",
    )

    pricing_force_refresh_on_startup: bool = Field(
        default=False,
        description="Whether to force pricing refresh immediately on startup",
    )

    # Observability tasks (migrated from ObservabilitySettings)
    pushgateway_enabled: bool = Field(
        default=False,
        description="Whether pushgateway metrics pushing task is enabled",
    )

    pushgateway_interval_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,  # Max 1 hour
        description="Interval in seconds between pushgateway metric pushes",
    )

    pushgateway_max_backoff_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=1800.0,  # Max 30 minutes
        description="Maximum backoff delay for failed pushgateway operations",
    )

    stats_printing_enabled: bool = Field(
        default=False,
        description="Whether stats printing task is enabled",
    )

    stats_printing_interval_seconds: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,  # Max 1 hour
        description="Interval in seconds between stats printing",
    )

    # Version checking task settings
    version_check_enabled: bool = Field(
        default=True,
        description="Whether version update checking is enabled",
    )

    version_check_interval_hours: int = Field(
        default=12,
        ge=1,
        le=168,  # Max 1 week
        description="Interval in hours between version checks",
    )

    version_check_startup_max_age_hours: float = Field(
        default=1.0,
        ge=0.1,
        le=24.0,
        description="Maximum age in hours since last check before running startup check",
    )

    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER__",
        case_sensitive=False,
    )
