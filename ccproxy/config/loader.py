"""Configuration file loader for ccproxy."""

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from ccproxy.config.discovery import find_toml_config_file
from ccproxy.config.settings import Settings


class ConfigurationError(Exception):
    """Configuration loading error."""

    pass


class ConfigLoader:
    """Load configuration from multiple sources."""

    def __init__(self) -> None:
        self._cached_config: dict[str, Any] | None = None

    def load(self, config_file: Path | None = None) -> Settings:
        """Load configuration from multiple sources.

        Priority: ENV > config file > defaults

        Args:
            config_file: Optional path to config file

        Returns:
            Settings instance with loaded configuration

        Raises:
            ConfigurationError: If config file is invalid or cannot be loaded
        """
        config_data = self._load_config_file(config_file)

        # Environment variables take precedence over config file
        return Settings(**config_data) if config_data else Settings()

    def _load_config_file(self, config_file: Path | None = None) -> dict[str, Any]:
        """Load configuration from file.

        Args:
            config_file: Optional path to config file

        Returns:
            Configuration dictionary

        Raises:
            ConfigurationError: If config file is invalid
        """
        if config_file is None:
            config_file = find_toml_config_file()

        if config_file is None or not config_file.exists():
            return {}

        try:
            if config_file.suffix.lower() in [".toml", ".tml"]:
                return self._load_toml_config(config_file)
            else:
                raise ConfigurationError(
                    f"Unsupported config file format: {config_file.suffix}. Only TOML (.toml) files are supported."
                )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load config file {config_file}: {e}"
            ) from e

    def _load_toml_config(self, config_file: Path) -> dict[str, Any]:
        """Load TOML configuration file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                raise ConfigurationError(
                    "TOML support not available. Install 'tomli' for Python < 3.11"
                ) from None

        with config_file.open("rb") as f:
            data = tomllib.load(f)
            return data if isinstance(data, dict) else {}

    def clear_cache(self) -> None:
        """Clear cached configuration."""
        self._cached_config = None


# Global config loader instance
config_loader = ConfigLoader()


def load_config(config_file: Path | None = None) -> Settings:
    """Load configuration using the global loader.

    Args:
        config_file: Optional path to config file

    Returns:
        Settings instance with loaded configuration
    """
    return config_loader.load(config_file)
