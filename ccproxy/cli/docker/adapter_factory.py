"""Docker adapter factory for CLI commands.

This module provides functions to create Docker adapters from CLI settings
and command-line arguments.
"""

import getpass
from pathlib import Path
from typing import Any

from ccproxy.config.settings import Settings
from ccproxy.docker import (
    DockerEnv,
    DockerPath,
    DockerUserContext,
    DockerVolume,
)


def _create_docker_adapter_from_settings(
    settings: Settings,
    docker_image: str | None = None,
    docker_env: list[str] | None = None,
    docker_volume: list[str] | None = None,
    docker_arg: list[str] | None = None,
    docker_home: str | None = None,
    docker_workspace: str | None = None,
    user_mapping_enabled: bool | None = None,
    user_uid: int | None = None,
    user_gid: int | None = None,
    command: list[str] | None = None,
    cmd_args: list[str] | None = None,
    **kwargs: Any,
) -> tuple[
    str,
    list[DockerVolume],
    DockerEnv,
    list[str] | None,
    DockerUserContext | None,
    list[str],
]:
    """Convert settings and overrides to Docker adapter parameters.

    Args:
        settings: Application settings
        docker_image: Override Docker image
        docker_env: Additional environment variables
        docker_volume: Additional volume mappings
        docker_arg: Additional Docker arguments
        docker_home: Override home directory
        docker_workspace: Override workspace directory
        user_mapping_enabled: Override user mapping setting
        user_uid: Override user ID
        user_gid: Override group ID
        command: Command to run in container
        cmd_args: Arguments for the command
        **kwargs: Additional keyword arguments (ignored)

    Returns:
        Tuple of (image, volumes, environment, command, user_context, additional_args)
    """
    docker_settings = settings.docker

    # Determine effective image
    image = docker_image or docker_settings.docker_image

    # Process volumes
    volumes: list[DockerVolume] = []

    # Add home/workspace volumes with effective directories
    home_dir = docker_home or docker_settings.docker_home_directory
    workspace_dir = docker_workspace or docker_settings.docker_workspace_directory

    if home_dir:
        volumes.append((str(Path(home_dir)), "/data/home"))
    if workspace_dir:
        volumes.append((str(Path(workspace_dir)), "/data/workspace"))

    # Add base volumes from settings
    for vol_str in docker_settings.docker_volumes:
        parts = vol_str.split(":", 2)
        if len(parts) >= 2:
            volumes.append((parts[0], parts[1]))

    # Add CLI override volumes
    if docker_volume:
        for vol_str in docker_volume:
            parts = vol_str.split(":", 2)
            if len(parts) >= 2:
                volumes.append((parts[0], parts[1]))

    # Process environment
    environment: DockerEnv = docker_settings.docker_environment.copy()

    # Add home/workspace environment variables
    if home_dir:
        environment["CLAUDE_HOME"] = "/data/home"
    if workspace_dir:
        environment["CLAUDE_WORKSPACE"] = "/data/workspace"

    # Add CLI override environment
    if docker_env:
        for env_var in docker_env:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                environment[key] = value

    # Create user context
    user_context = None
    effective_mapping_enabled = (
        user_mapping_enabled
        if user_mapping_enabled is not None
        else docker_settings.user_mapping_enabled
    )

    if effective_mapping_enabled:
        effective_uid = user_uid if user_uid is not None else docker_settings.user_uid
        effective_gid = user_gid if user_gid is not None else docker_settings.user_gid

        if effective_uid is not None and effective_gid is not None:
            # Create DockerPath instances for user context
            home_path = None
            workspace_path = None

            if home_dir:
                home_path = DockerPath(
                    host_path=Path(home_dir), container_path="/data/home"
                )
            if workspace_dir:
                workspace_path = DockerPath(
                    host_path=Path(workspace_dir), container_path="/data/workspace"
                )

            # Use a default username if not available
            username = getpass.getuser()

            user_context = DockerUserContext(
                uid=effective_uid,
                gid=effective_gid,
                username=username,
                home_path=home_path,
                workspace_path=workspace_path,
            )

    # Build command
    final_command = None
    if command:
        final_command = command.copy()
        if cmd_args:
            final_command.extend(cmd_args)

    # Additional Docker arguments
    additional_args = docker_settings.docker_additional_args.copy()
    if docker_arg:
        additional_args.extend(docker_arg)

    return image, volumes, environment, final_command, user_context, additional_args
