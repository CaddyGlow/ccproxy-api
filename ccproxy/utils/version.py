import re


def parse_version(version_string: str) -> tuple[int, int, int, str]:
    """
    Parse version string into components.

    Handles various formats:
    - 1.2.3
    - 1.2.3-dev
    - 1.2.3.dev59+g1624e1e.d19800101
    - 0.1.dev59+g1624e1e.d19800101
    """
    # Clean up setuptools-scm dev versions
    clean_version = re.sub(r"\.dev\d+\+.*", "", version_string)

    # Handle dev versions without patch number
    if ".dev" in version_string:
        base_version = version_string.split(".dev")[0]
        parts = base_version.split(".")
        if len(parts) == 2:
            # 0.1.dev59 -> 0.1.0-dev
            major, minor = int(parts[0]), int(parts[1])
            patch = 0
            suffix = "dev"
        else:
            # 1.2.3.dev59 -> 1.2.3-dev
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            suffix = "dev"
    else:
        # Regular version
        parts = clean_version.split(".")
        if len(parts) < 3:
            parts.extend(["0"] * (3 - len(parts)))

        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        suffix = ""

    return major, minor, patch, suffix


def format_version(version: str, level: str) -> str:
    major, minor, patch, suffix = parse_version(version)

    """Format version according to specified level."""
    base_version = f"{major}.{minor}.{patch}"

    if level == "major":
        return str(major)
    elif level == "minor":
        return f"{major}.{minor}"
    elif level == "patch" or level == "full":
        if suffix:
            return f"{base_version}-{suffix}"
        return base_version
    elif level == "docker":
        # Docker-compatible version (no + characters)
        if suffix:
            return f"{base_version}-{suffix}"
        return base_version
    elif level == "npm":
        # NPM-compatible version
        if suffix:
            return f"{base_version}-{suffix}.0"
        return base_version
    elif level == "python":
        # Python-compatible version
        if suffix:
            return f"{base_version}.{suffix}0"
        return base_version
    else:
        raise ValueError(f"Unknown version level: {level}")


def get_next_minor_version(version: str) -> str:
    """
    Get the next minor version.

    Examples:
    - 1.2.3 -> 1.3.0
    - 1.2.3-dev -> 1.3.0
    - 0.1.dev59+g1624e1e -> 0.2.0
    """
    major, minor, _, _ = parse_version(version)
    return f"{major}.{minor + 1}.0"


def get_next_major_version(version: str) -> str:
    """
    Get the next major version.

    Examples:
    - 1.2.3 -> 2.0.0
    - 1.2.3-dev -> 2.0.0
    - 0.1.dev59+g1624e1e -> 1.0.0
    """
    major, _, _, _ = parse_version(version)
    return f"{major + 1}.0.0"
