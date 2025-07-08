"""Docker validation utilities for Claude Code Proxy API."""

import os
from pathlib import Path


def validate_volume_format(volume: str) -> str:
    """Validate and normalize volume mount format.

    Args:
        volume: Volume string in 'host:container[:options]' format

    Returns:
        Normalized volume string with absolute host path

    Raises:
        ValueError: If volume format is invalid or host path doesn't exist
    """
    if ":" not in volume:
        raise ValueError(
            f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
        )
    parts = volume.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
        )

    # Check for empty host path
    if not parts[0].strip():
        raise ValueError(
            f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
        )

    # Convert relative paths to absolute
    host_path = os.path.expandvars(parts[0])
    path_obj = Path(host_path)

    # If it's a relative path, convert to absolute
    if not path_obj.is_absolute():
        host_path = str(path_obj.resolve())

    # Check if the absolute path exists
    if not Path(host_path).exists():
        raise ValueError(f"Host path does not exist: '{host_path}'")

    # Return normalized volume string
    parts[0] = host_path
    return ":".join(parts)


def validate_host_path(path: str) -> str:
    """Validate and normalize host path format.

    Args:
        path: Host path string

    Returns:
        Normalized host path as absolute path

    Raises:
        ValueError: If path is invalid
    """
    # Expand environment variables
    expanded_path = os.path.expandvars(path)
    path_obj = Path(expanded_path)

    # If it's a relative path, convert to absolute
    if not path_obj.is_absolute():
        path = str(path_obj.resolve())
    else:
        path = expanded_path

    return path


def validate_environment_variable(env_var: str) -> tuple[str, str]:
    """Validate and parse environment variable format.

    Args:
        env_var: Environment variable in 'KEY=VALUE' format

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If environment variable format is invalid
    """
    if "=" not in env_var:
        raise ValueError(
            f"Invalid environment variable format: '{env_var}'. Expected KEY=VALUE"
        )
    key, value = env_var.split("=", 1)
    return key, value


def validate_docker_volumes(volumes: list[str]) -> list[str]:
    """Validate a list of Docker volume mount formats.

    Args:
        volumes: List of volume strings in 'host:container[:options]' format

    Returns:
        List of normalized volume strings with absolute host paths

    Raises:
        ValueError: If any volume format is invalid or host path doesn't exist
    """
    validated_volumes = []
    for volume in volumes:
        validated_volumes.append(validate_volume_format(volume))
    return validated_volumes
