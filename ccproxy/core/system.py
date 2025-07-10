import os
from pathlib import Path


def get_xdg_config_home() -> Path:
    """Get the XDG_CONFIG_HOME directory.

    Returns:
        Path to the XDG config directory. Falls back to ~/.config if not set.
    """
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home)
    return Path.home() / ".config"


def get_xdg_data_home() -> Path:
    """Get the XDG_DATA_HOME directory.

    Returns:
        Path to the XDG data directory. Falls back to ~/.local/share if not set.
    """
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home)
    return Path.home() / ".local" / "share"


def get_xdg_cache_home() -> Path:
    """Get the XDG_CACHE_HOME directory.

    Returns:
        Path to the XDG cache directory. Falls back to ~/.cache if not set.
    """
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return Path.home() / ".cache"
