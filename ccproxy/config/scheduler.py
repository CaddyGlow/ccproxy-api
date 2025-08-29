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
        description="Whether pricing cache update task is enabled. Enabled by default for privacy - downloads from GitHub when enabled",
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

    # Pushgateway settings removed - functionality moved to metrics plugin
    # The metrics plugin now manages its own pushgateway configuration

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
        description="Whether version update checking is enabled. Enabled by default for privacy - checks GitHub API when enabled",
    )

    version_check_interval_hours: int = Field(
        default=6,
        ge=1,
        le=168,  # Max 1 week
        description="Interval in hours between version checks",
    )

    version_check_cache_ttl_hours: float = Field(
        default=6,
        ge=0.1,
        le=24.0,
        description="Maximum age in hours since last check version check",
    )

    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER__",
        case_sensitive=False,
    )
