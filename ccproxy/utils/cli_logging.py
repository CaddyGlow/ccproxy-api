"""Dynamic CLI logging utilities."""

from typing import Any

import structlog

from .binary_resolver import CLIInfo


logger = structlog.get_logger(__name__)


def log_cli_info(cli_info_dict: dict[str, CLIInfo], context: str = "plugin") -> None:
    """Log CLI information dynamically for each CLI found.

    Args:
        cli_info_dict: Dictionary of CLI name -> CLIInfo
        context: Context for logging (e.g., "plugin", "startup", "detection")
    """
    for cli_name, cli_info in cli_info_dict.items():
        if cli_info["is_available"]:
            logger.debug(
                f"{context}_cli_available",
                cli_name=cli_name,
                version=cli_info["version"],
                source=cli_info["source"],
                path=cli_info["path"],
                command=cli_info["command"],
                package_manager=cli_info["package_manager"],
            )
        else:
            logger.warning(
                f"{context}_cli_unavailable",
                cli_name=cli_name,
                expected_version=cli_info["version"],
            )


def log_plugin_summary(summary: dict[str, Any], plugin_name: str) -> None:
    """Log plugin summary with dynamic CLI information.

    Args:
        summary: Plugin summary dictionary
        plugin_name: Name of the plugin
    """
    # Log basic plugin info
    basic_info = {k: v for k, v in summary.items() if k != "cli_info"}
    logger.debug(
        "plugin_summary",
        plugin_name=plugin_name,
        **basic_info,
    )

    # Log CLI info dynamically if present
    if "cli_info" in summary:
        log_cli_info(summary["cli_info"], f"{plugin_name}_plugin")


def format_cli_info_for_display(cli_info: CLIInfo) -> dict[str, str]:
    """Format CLI info for human-readable display.

    Args:
        cli_info: CLI information dictionary

    Returns:
        Formatted dictionary for display
    """
    if not cli_info["is_available"]:
        return {
            "status": "unavailable",
            "name": cli_info["name"],
        }

    display_info = {
        "status": "available",
        "name": cli_info["name"],
        "version": cli_info["version"] or "unknown",
        "source": cli_info["source"],
    }

    if cli_info["source"] == "path":
        display_info["path"] = cli_info["path"] or "unknown"
    elif cli_info["source"] == "package_manager":
        display_info["package_manager"] = cli_info["package_manager"] or "unknown"
        display_info["command"] = " ".join(cli_info["command"])

    return display_info


def create_cli_summary_table(cli_info_dict: dict[str, CLIInfo]) -> list[dict[str, str]]:
    """Create a table-ready summary of all CLI information.

    Args:
        cli_info_dict: Dictionary of CLI name -> CLIInfo

    Returns:
        List of formatted CLI info for table display
    """
    return [
        format_cli_info_for_display(cli_info) for cli_info in cli_info_dict.values()
    ]
