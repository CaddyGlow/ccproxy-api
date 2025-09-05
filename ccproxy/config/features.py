"""Feature flag configuration for safe deployment of new functionality."""

from pydantic_settings import BaseSettings


class FeatureSettings(BaseSettings):
    """Feature flags for safe deployment of new functionality."""

    # Format adapter manifest system
    manifest_format_adapters: bool = False

    # Migration control
    deprecate_manual_format_setup: bool = False

    class Config:
        env_prefix = "FEATURES__"
