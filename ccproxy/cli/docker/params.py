"""Shared Docker parameter definitions for Typer CLI commands.

This module provides reusable Typer Option definitions for Docker-related
parameters that are used across multiple CLI commands, eliminating duplication.
"""

from typing import Any

import typer


# Docker parameter validation functions moved here to avoid utils dependency


def parse_docker_env(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str] | None
) -> list[str]:
    """Parse Docker environment variable string."""
    if not value:
        return []

    parsed = []
    for env_str in value:
        if not env_str or env_str == "[]":
            raise typer.BadParameter(
                f"Invalid env format: {env_str}. Expected KEY=VALUE"
            )
        if "=" not in env_str:
            raise typer.BadParameter(
                f"Invalid env format: {env_str}. Expected KEY=VALUE"
            )
        parsed.append(env_str)

    return parsed


def parse_docker_volume(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str] | None
) -> list[str]:
    """Parse Docker volume string."""
    if not value:
        return []

    # Import the validation function from config
    from ccproxy.config.docker_settings import validate_volume_format

    parsed = []
    for volume_str in value:
        if not volume_str:
            continue
        try:
            validated_volume = validate_volume_format(volume_str)
            parsed.append(validated_volume)
        except ValueError as e:
            raise typer.BadParameter(str(e)) from e

    return parsed


def validate_docker_arg(
    ctx: typer.Context, param: typer.CallbackParam, value: list[str] | None
) -> list[str]:
    """Validate Docker argument."""
    if not value:
        return []

    # Basic validation - ensure arguments don't contain dangerous patterns
    validated = []
    for arg in value:
        if not arg:
            continue
        # Basic validation - just return the arg for now
        validated.append(arg)

    return validated


def validate_docker_home(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate Docker home directory."""
    if value is None:
        return None

    from pathlib import Path

    from ccproxy.config.docker_settings import validate_host_path

    try:
        return validate_host_path(value)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def validate_docker_image(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate Docker image name."""
    if value is None:
        return None

    if not value:
        raise typer.BadParameter("Docker image cannot be empty")

    # Basic validation - no spaces allowed in image names
    if " " in value:
        raise typer.BadParameter(f"Docker image name cannot contain spaces: {value}")

    return value


def validate_docker_workspace(
    ctx: typer.Context, param: typer.CallbackParam, value: str | None
) -> str | None:
    """Validate Docker workspace directory."""
    if value is None:
        return None

    from pathlib import Path

    from ccproxy.config.docker_settings import validate_host_path

    try:
        return validate_host_path(value)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def validate_user_gid(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate user GID."""
    if value is None:
        return None

    if value < 0:
        raise typer.BadParameter("GID must be non-negative")

    return value


def validate_user_uid(
    ctx: typer.Context, param: typer.CallbackParam, value: int | None
) -> int | None:
    """Validate user UID."""
    if value is None:
        return None

    if value < 0:
        raise typer.BadParameter("UID must be non-negative")

    return value


def docker_image_option() -> Any:
    """Docker image parameter."""
    return typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
    )


def docker_env_option() -> Any:
    """Docker environment variables parameter."""
    return typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
    )


def docker_volume_option() -> Any:
    """Docker volume mounts parameter."""
    return typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
    )


def docker_arg_option() -> Any:
    """Docker arguments parameter."""
    return typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
    )


def docker_home_option() -> Any:
    """Docker home directory parameter."""
    return typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
    )


def docker_workspace_option() -> Any:
    """Docker workspace directory parameter."""
    return typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
    )


def user_mapping_option() -> Any:
    """User mapping parameter."""
    return typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    )


def user_uid_option() -> Any:
    """User UID parameter."""
    return typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    )


def user_gid_option() -> Any:
    """User GID parameter."""
    return typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
    )


class DockerOptions:
    """Container for all Docker-related Typer options.

    This class provides a convenient way to include all Docker-related
    options in a command using typed attributes.
    """

    def __init__(
        self,
        docker_image: str | None = None,
        docker_env: list[str] | None = None,
        docker_volume: list[str] | None = None,
        docker_arg: list[str] | None = None,
        docker_home: str | None = None,
        docker_workspace: str | None = None,
        user_mapping_enabled: bool | None = None,
        user_uid: int | None = None,
        user_gid: int | None = None,
    ):
        """Initialize Docker options.

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
        """
        self.docker_image = docker_image
        self.docker_env = docker_env or []
        self.docker_volume = docker_volume or []
        self.docker_arg = docker_arg or []
        self.docker_home = docker_home
        self.docker_workspace = docker_workspace
        self.user_mapping_enabled = user_mapping_enabled
        self.user_uid = user_uid
        self.user_gid = user_gid
