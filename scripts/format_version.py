#!/usr/bin/env python3
"""
Version formatting script for CI/CD pipelines.

Extracts version from package __version__ and formats it for different semver levels.
"""

import argparse
import sys
from pathlib import Path


# Only add to sys.path if claude_code_proxy module is not available
try:
    import claude_code_proxy  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_code_proxy import __version__
from claude_code_proxy.utils import format_version, parse_version


def main() -> None:
    """Main function to extract and format version."""
    parser = argparse.ArgumentParser(
        description="Version formatting script for CI/CD pipelines",
        epilog="Extracts version from package __version__ and formats it for different semver levels.",
    )
    parser.add_argument(
        "level",
        choices=["major", "minor", "patch", "full", "docker", "npm", "python"],
        help="Version level to format",
    )

    args = parser.parse_args()
    level = args.level.lower()

    try:
        # Format according to requested level
        formatted_version = format_version(__version__, level)

        print(formatted_version)

    except (ImportError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
