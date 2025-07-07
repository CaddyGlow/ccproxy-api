"""Shared Docker parameter definitions for Typer CLI commands.

This module provides reusable Typer Option definitions for Docker-related
parameters that are used across multiple CLI commands, eliminating duplication.
"""

import typer

from claude_code_proxy.utils.docker_params import (
    parse_docker_env,
    parse_docker_volume,
    validate_docker_arg,
    validate_docker_home,
    validate_docker_image,
    validate_docker_workspace,
    validate_user_gid,
    validate_user_uid,
)


def docker_image_option(
    value: str | None = typer.Option(
        None,
        "--docker-image",
        help="Docker image to use (overrides config)",
        callback=validate_docker_image,
    ),
) -> str | None:
    """Docker image parameter."""
    return value


def docker_env_option(
    value: list[str] = typer.Option(
        [],
        "--docker-env",
        help="Environment variables to pass to Docker (KEY=VALUE format, can be used multiple times)",
        callback=parse_docker_env,
    ),
) -> list[str]:
    """Docker environment variables parameter."""
    return value


def docker_volume_option(
    value: list[str] = typer.Option(
        [],
        "--docker-volume",
        help="Volume mounts to add (host:container[:options] format, can be used multiple times)",
        callback=parse_docker_volume,
    ),
) -> list[str]:
    """Docker volume mounts parameter."""
    return value


def docker_arg_option(
    value: list[str] = typer.Option(
        [],
        "--docker-arg",
        help="Additional Docker run arguments (can be used multiple times)",
        callback=validate_docker_arg,
    ),
) -> list[str]:
    """Docker arguments parameter."""
    return value


def docker_home_option(
    value: str | None = typer.Option(
        None,
        "--docker-home",
        help="Home directory inside Docker container (overrides config)",
        callback=validate_docker_home,
    ),
) -> str | None:
    """Docker home directory parameter."""
    return value


def docker_workspace_option(
    value: str | None = typer.Option(
        None,
        "--docker-workspace",
        help="Workspace directory inside Docker container (overrides config)",
        callback=validate_docker_workspace,
    ),
) -> str | None:
    """Docker workspace directory parameter."""
    return value


def user_mapping_option(
    value: bool | None = typer.Option(
        None,
        "--user-mapping/--no-user-mapping",
        help="Enable/disable UID/GID mapping (overrides config)",
    ),
) -> bool | None:
    """User mapping parameter."""
    return value


def user_uid_option(
    value: int | None = typer.Option(
        None,
        "--user-uid",
        help="User ID to run container as (overrides config)",
        min=0,
        callback=validate_user_uid,
    ),
) -> int | None:
    """User UID parameter."""
    return value


def user_gid_option(
    value: int | None = typer.Option(
        None,
        "--user-gid",
        help="Group ID to run container as (overrides config)",
        min=0,
        callback=validate_user_gid,
    ),
) -> int | None:
    """User GID parameter."""
    return value


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
