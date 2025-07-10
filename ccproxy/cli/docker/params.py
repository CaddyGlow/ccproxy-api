"""Shared Docker parameter definitions for Typer CLI commands.

This module provides reusable Typer Option definitions for Docker-related
parameters that are used across multiple CLI commands, eliminating duplication.
"""

import typer


# Docker parameter validation functions moved here to avoid utils dependency


def parse_docker_env(env_str: str) -> tuple[str, str]:
    """Parse Docker environment variable string."""
    if not env_str or env_str == "[]":
        raise ValueError(f"Invalid env format: {env_str}. Expected KEY=VALUE")
    if "=" not in env_str:
        raise ValueError(f"Invalid env format: {env_str}. Expected KEY=VALUE")
    key, value = env_str.split("=", 1)
    return key, value


def parse_docker_volume(volume_str: str) -> tuple[str, str]:
    """Parse Docker volume string."""
    parts = volume_str.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid volume format: {volume_str}. Expected host:container"
        )
    return parts[0], parts[1]


def validate_docker_arg(arg: str) -> str:
    """Validate Docker argument."""
    return arg


def validate_docker_home(path: str) -> str:
    """Validate Docker home directory."""
    from pathlib import Path

    return str(Path(path).resolve())


def validate_docker_image(image: str) -> str:
    """Validate Docker image name."""
    return image


def validate_docker_workspace(path: str) -> str:
    """Validate Docker workspace directory."""
    from pathlib import Path

    return str(Path(path).resolve())


def validate_user_gid(gid: str) -> str:
    """Validate user GID."""
    return gid


def validate_user_uid(uid: str) -> str:
    """Validate user UID."""
    return uid


def docker_image_option():
    """Docker image parameter."""
    return typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
    )


def docker_env_option():
    """Docker environment variables parameter."""
    return typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
    )


def docker_volume_option():
    """Docker volume mounts parameter."""
    return typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
    )


def docker_arg_option():
    """Docker arguments parameter."""
    return typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
    )


def docker_home_option():
    """Docker home directory parameter."""
    return typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
    )


def docker_workspace_option():
    """Docker workspace directory parameter."""
    return typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
    )


def user_mapping_option():
    """User mapping parameter."""
    return typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    )


def user_uid_option():
    """User UID parameter."""
    return typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
    )


def user_gid_option():
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
