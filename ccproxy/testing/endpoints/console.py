"""Utilities for producing colorized terminal output."""

from __future__ import annotations


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"


def colored_header(title: str) -> str:
    """Create a simple colored header for section boundaries."""
    return f"\n{Colors.BOLD}{Colors.CYAN}== {title} =={Colors.RESET}\n"


def colored_success(text: str) -> str:
    """Color text as success (green)."""
    return f"{Colors.GREEN}{text}{Colors.RESET}"


def colored_error(text: str) -> str:
    """Color text as error (red)."""
    return f"{Colors.RED}{text}{Colors.RESET}"


def colored_info(text: str) -> str:
    """Color text as info (blue)."""
    return f"{Colors.BLUE}{text}{Colors.RESET}"


def colored_progress(text: str) -> str:
    """Color progress messages with a lighter cyan tone."""
    return f"{Colors.CYAN}{text}{Colors.RESET}"


def colored_warning(text: str) -> str:
    """Color text as warning (yellow)."""
    return f"{Colors.YELLOW}{text}{Colors.RESET}"


__all__ = [
    "Colors",
    "colored_header",
    "colored_success",
    "colored_error",
    "colored_info",
    "colored_progress",
    "colored_warning",
]
