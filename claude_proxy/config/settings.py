"""Settings configuration for Claude Proxy API Server."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings for the Claude Proxy API Server.

    Settings are loaded from environment variables and .env files.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required settings
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude access",
        min_length=1,
    )

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )

    port: int = Field(
        default=8000,
        description="Server port number",
        ge=1,
        le=65535,
    )

    # Logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Optional server settings
    workers: int = Field(
        default=1,
        description="Number of worker processes",
        ge=1,
        le=32,
    )

    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )

    # Rate limiting settings
    rate_limit_requests: int = Field(
        default=100,
        description="Rate limit: maximum requests per minute",
        ge=1,
    )

    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds",
        ge=1,
    )

    # Request timeout settings
    request_timeout: int = Field(
        default=300,
        description="Request timeout in seconds",
        ge=1,
        le=3600,
    )

    # Security settings
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )

    # Configuration file path
    config_file: Path | None = Field(
        default=None,
        description="Path to JSON configuration file",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        return v.upper()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("config_file", mode="before")
    @classmethod
    def validate_config_file(cls, v: str | Path | None) -> Path | None:
        """Convert string path to Path object."""
        if v is None:
            return None
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def server_url(self) -> str:
        """Get the complete server URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.reload or self.log_level == "DEBUG"

    def model_dump_safe(self) -> dict[str, Any]:
        """
        Dump model data with sensitive information masked.

        Returns:
            dict: Configuration with sensitive data masked
        """
        data = self.model_dump()
        if "anthropic_api_key" in data:
            data["anthropic_api_key"] = "***MASKED***"
        return data


def get_settings() -> Settings:
    """Get the global settings instance."""
    try:
        return Settings()
    except Exception as e:
        # If settings can't be loaded (e.g., missing API key),
        # this will be handled by the caller
        raise ValueError(f"Configuration error: {e}") from e
