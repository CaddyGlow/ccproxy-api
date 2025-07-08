"""Docker parameter type definitions and converters for CLI options.

This module provides validation and conversion functions for Docker-related
CLI parameters, ensuring proper formatting and type safety.
"""

import os
from pathlib import Path
from typing import Any

import typer


def validate_docker_image(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate Docker image name format.

    Args:
        ctx: Typer context
        param: Parameter info
        value: The Docker image name to validate

    Returns:
        Validated Docker image name or None

    Raises:
        typer.BadParameter: If the image name format is invalid
    """
    if value is None:
        return None

    # Basic validation - ensure it's not empty or whitespace
    if not value.strip():
        raise typer.BadParameter("Docker image name cannot be empty")

    # Docker image names can contain lowercase letters, digits, periods, dashes, underscores, and slashes
    # We're being permissive here as Docker itself will validate more strictly
    if value.count(" ") > 0:
        raise typer.BadParameter("Docker image name cannot contain spaces")

    return value.strip()


def parse_docker_env(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str]
) -> list[str]:
    """Parse and validate Docker environment variable format.

    Args:
        ctx: Typer context
        param: Parameter info
        value: List of environment variables in KEY=VALUE format

    Returns:
        Validated list of environment variables

    Raises:
        typer.BadParameter: If any environment variable format is invalid
    """
    if not value:
        return []

    validated = []
    for env_var in value:
        if "=" not in env_var:
            raise typer.BadParameter(
                f"Invalid environment variable format: '{env_var}'. Expected KEY=VALUE"
            )
        key, val = env_var.split("=", 1)
        if not key.strip():
            raise typer.BadParameter(
                f"Invalid environment variable: '{env_var}'. Key cannot be empty"
            )
        validated.append(env_var)

    return validated


def parse_docker_volume(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str]
) -> list[str]:
    """Parse and validate Docker volume mount format.

    Args:
        ctx: Typer context
        param: Parameter info
        value: List of volume mounts in host:container[:options] format

    Returns:
        Validated list of volume mounts with normalized paths

    Raises:
        typer.BadParameter: If any volume format is invalid
    """
    if not value:
        return []

    validated = []
    for volume in value:
        if ":" not in volume:
            raise typer.BadParameter(
                f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
            )

        parts = volume.split(":", 2)
        if len(parts) < 2:
            raise typer.BadParameter(
                f"Invalid volume format: '{volume}'. Expected 'host:container[:options]'"
            )

        # Check for empty host path
        if not parts[0].strip():
            raise typer.BadParameter(
                f"Invalid volume format: '{volume}'. Host path cannot be empty"
            )

        # Expand environment variables in host path
        host_path = os.path.expandvars(parts[0])
        path_obj = Path(host_path)

        # Convert relative paths to absolute
        if not path_obj.is_absolute():
            host_path = str(path_obj.resolve())

        # Check if the absolute path exists
        if not Path(host_path).exists():
            raise typer.BadParameter(f"Host path does not exist: '{host_path}'")

        # Reconstruct the volume string with normalized path
        parts[0] = host_path
        validated.append(":".join(parts))

    return validated


def validate_docker_arg(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str]
) -> list[str]:
    """Validate additional Docker run arguments.

    Args:
        ctx: Typer context
        param: Parameter info
        value: List of additional Docker arguments

    Returns:
        Validated list of Docker arguments

    Raises:
        typer.BadParameter: If any argument is invalid
    """
    if not value:
        return []

    # Basic validation - ensure no empty arguments
    for arg in value:
        if not arg.strip():
            raise typer.BadParameter("Docker argument cannot be empty")

    return value


def validate_docker_home(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate and normalize Docker home directory path.

    Args:
        ctx: Typer context
        param: Parameter info
        value: The home directory path

    Returns:
        Normalized absolute path or None

    Raises:
        typer.BadParameter: If the path format is invalid
    """
    if value is None:
        return None

    # Expand environment variables
    expanded_path = os.path.expandvars(value)
    path_obj = Path(expanded_path)

    # Convert relative paths to absolute
    if not path_obj.is_absolute():
        return str(path_obj.resolve())

    return expanded_path


def validate_docker_workspace(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate and normalize Docker workspace directory path.

    Args:
        ctx: Typer context
        param: Parameter info
        value: The workspace directory path

    Returns:
        Normalized absolute path or None

    Raises:
        typer.BadParameter: If the path format is invalid
    """
    if value is None:
        return None

    # Expand environment variables
    expanded_path = os.path.expandvars(value)
    path_obj = Path(expanded_path)

    # Convert relative paths to absolute
    if not path_obj.is_absolute():
        return str(path_obj.resolve())

    return expanded_path


def validate_user_uid(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate user ID value.

    Args:
        ctx: Typer context
        param: Parameter info
        value: The user ID

    Returns:
        Validated user ID or None

    Raises:
        typer.BadParameter: If the UID is invalid
    """
    if value is None:
        return None

    if value < 0:
        raise typer.BadParameter("User ID must be non-negative")

    return value


def validate_user_gid(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate group ID value.

    Args:
        ctx: Typer context
        param: Parameter info
        value: The group ID

    Returns:
        Validated group ID or None

    Raises:
        typer.BadParameter: If the GID is invalid
    """
    if value is None:
        return None

    if value < 0:
        raise typer.BadParameter("Group ID must be non-negative")

    return value


def extract_docker_params(
    docker_image: str | None = None,
    docker_env: list[str] | None = None,
    docker_volume: list[str] | None = None,
    docker_arg: list[str] | None = None,
    docker_home: str | None = None,
    docker_workspace: str | None = None,
    user_mapping_enabled: bool | None = None,
    user_uid: int | None = None,
    user_gid: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Extract Docker-related parameters from keyword arguments.

    This function is useful for collecting all Docker-related parameters
    from a larger set of CLI arguments.

    Args:
        docker_image: Docker image to use
        docker_env: Environment variables list
        docker_volume: Volume mounts list
        docker_arg: Additional Docker arguments
        docker_home: Home directory path
        docker_workspace: Workspace directory path
        user_mapping_enabled: User mapping flag
        user_uid: User ID
        user_gid: Group ID
        **kwargs: Other arguments (ignored)

    Returns:
        Dictionary containing only Docker-related parameters
    """
    return {
        "docker_image": docker_image,
        "docker_env": docker_env or [],
        "docker_volume": docker_volume or [],
        "docker_arg": docker_arg or [],
        "docker_home": docker_home,
        "docker_workspace": docker_workspace,
        "user_mapping_enabled": user_mapping_enabled,
        "user_uid": user_uid,
        "user_gid": user_gid,
    }


def merge_docker_params(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge Docker parameter dictionaries with proper list handling.

    Args:
        base: Base parameters (e.g., from config)
        overrides: Override parameters (e.g., from CLI)

    Returns:
        Merged parameters dictionary
    """
    result = base.copy()

    for key, value in overrides.items():
        if value is not None:
            if key in ["docker_env", "docker_volume", "docker_arg"] and isinstance(
                value, list
            ):
                # For list parameters, extend rather than replace
                base_list = result.get(key, [])
                if isinstance(base_list, list):
                    result[key] = base_list + value
                else:
                    result[key] = value
            else:
                # For other parameters, override
                result[key] = value

    return result
