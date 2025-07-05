"""XDG Base Directory Specification utilities."""

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


def get_ccproxy_config_dir() -> Path:
    """Get the ccproxy configuration directory.

    Returns:
        Path to the ccproxy configuration directory within XDG_CONFIG_HOME.
    """
    return get_xdg_config_home() / "ccproxy"


def get_claude_cli_config_dir() -> Path:
    """Get the Claude CLI configuration directory.

    Returns:
        Path to the Claude CLI configuration directory within XDG_CONFIG_HOME.
    """
    return get_xdg_config_home() / "claude"


def get_claude_docker_home_dir() -> Path:
    """Get the Claude Docker home directory.

    Returns:
        Path to the Claude Docker home directory within XDG_DATA_HOME.
    """
    return get_ccproxy_config_dir() / "home"
