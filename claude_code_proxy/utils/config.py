"""Configuration file discovery utilities."""

import os
from pathlib import Path

from .xdg import get_ccproxy_config_dir


def find_git_root(path: Path | None = None) -> Path | None:
    """Find the root directory of a git repository.

    Args:
        path: Starting path to search from. Defaults to current directory.

    Returns:
        Path to the git root directory, or None if not in a git repository.
    """
    if path is None:
        path = Path.cwd()

    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def find_toml_config_file() -> Path | None:
    """Find the TOML configuration file for ccproxy.

    Searches in the following order:
    1. .ccproxy.toml in current directory
    2. ccproxy.toml in git repository root (if in a git repo)
    3. config.toml in XDG_CONFIG_HOME/ccproxy/

    Returns:
        Path to the first found configuration file, or None if not found.
    """
    # 1. Check for .ccproxy.toml in current directory
    current_dir_config = Path.cwd() / ".ccproxy.toml"
    if current_dir_config.exists():
        return current_dir_config

    # 2. Check for ccproxy.toml in git repository root
    git_root = find_git_root()
    if git_root:
        repo_config = git_root / "ccproxy.toml"
        if repo_config.exists():
            return repo_config

    # 3. Check for config.toml in XDG_CONFIG_HOME/ccproxy/
    xdg_config = get_ccproxy_config_dir() / "config.toml"
    if xdg_config.exists():
        return xdg_config

    return None


def create_default_config_dir() -> Path:
    """Create the default configuration directory if it doesn't exist.

    Returns:
        Path to the ccproxy configuration directory.
    """
    config_dir = get_ccproxy_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir
