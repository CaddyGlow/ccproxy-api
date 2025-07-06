"""Utility modules for Claude proxy."""

from .config import (
    create_default_config_dir,
    find_git_root,
    find_toml_config_file,
)
from .helper import merge_claude_code_options
from .schema import (
    generate_json_schema,
    generate_schema_files,
    generate_taplo_config,
    save_schema_file,
    validate_toml_with_schema,
)
from .version import format_version, parse_version
from .xdg import (
    get_ccproxy_config_dir,
    get_claude_cli_config_dir,
    get_xdg_cache_home,
    get_xdg_config_home,
    get_xdg_data_home,
)


__all__ = [
    "create_default_config_dir",
    "find_git_root",
    "find_toml_config_file",
    "generate_json_schema",
    "generate_schema_files",
    "generate_taplo_config",
    "get_ccproxy_config_dir",
    "get_claude_cli_config_dir",
    "get_xdg_cache_home",
    "get_xdg_config_home",
    "get_xdg_data_home",
    "merge_claude_code_options",
    "save_schema_file",
    "validate_toml_with_schema",
    "format_version",
    "parse_version",
]
