"""Pricing configuration settings."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from ccproxy.core.system import get_xdg_cache_home


class PricingSettings(BaseSettings):
    """
    Configuration settings for the pricing system.

    Controls pricing cache behavior, data sources, and update mechanisms.
    Settings can be configured via environment variables with PRICING__ prefix.
    """

    # Cache settings
    cache_dir: Path = Field(
        default_factory=lambda: get_xdg_cache_home() / "ccproxy",
        description="Directory for pricing cache files (defaults to XDG_CACHE_HOME/ccproxy)",
    )

    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # Max 1 week
        description="Hours before pricing cache expires",
    )

    # Data source settings
    source_url: str = Field(
        default="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        description="URL to download pricing data from",
    )

    download_timeout: int = Field(
        default=30,
        ge=1,
        le=300,  # Max 5 minutes
        description="Request timeout in seconds for downloading pricing data",
    )

    # Update behavior settings
    auto_update: bool = Field(
        default=True,
        description="Whether to automatically update stale cache",
    )

    fallback_to_embedded: bool = Field(
        default=True,
        description="Whether to fallback to embedded pricing on failure",
    )

    # Memory cache settings
    memory_cache_ttl: int = Field(
        default=300,
        ge=1,
        le=3600,  # Max 1 hour
        description="Time to live for in-memory pricing cache in seconds",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def validate_cache_dir(cls, v: str | Path | None) -> Path:
        """Validate and convert cache directory path."""
        if v is None:
            return get_xdg_cache_home() / "ccproxy"
        if isinstance(v, str):
            if v.startswith("~/"):
                return Path(v).expanduser()
            return Path(v)
        return v

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v: str) -> str:
        """Validate source URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("Source URL must start with http:// or https://")
        return v

    class Config:
        """Pydantic configuration."""

        env_prefix = "PRICING__"
        case_sensitive = False
