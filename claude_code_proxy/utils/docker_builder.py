"""Docker command builder utility for Claude Proxy."""

import os
from pathlib import Path
from typing import Any

from claude_code_proxy.utils.docker_validation import (
    validate_environment_variable,
    validate_host_path,
    validate_volume_format,
)
from claude_code_proxy.utils.logging import get_logger

from ..config.settings import DockerSettings


logger = get_logger(__name__)


class DockerCommandBuilder:
    """Builds Docker commands with configurable settings and CLI overrides."""

    def __init__(self, docker_settings: DockerSettings) -> None:
        """Initialize with Docker settings from configuration.

        Args:
            docker_settings: Docker configuration from settings
        """
        self.settings = docker_settings

    def build_command(
        self,
        *,
        docker_image: str | None = None,
        docker_env: list[str] | None = None,
        docker_volume: list[str] | None = None,
        docker_workdir: str | None = None,
        docker_arg: list[str] | None = None,
        docker_home: str | None = None,
        docker_workspace: str | None = None,
        user_mapping_enabled: bool | None = None,
        user_uid: int | None = None,
        user_gid: int | None = None,
        entrypoint: str | None = None,
        command: list[str] | None = None,
        cmd_args: list[str] | None = None,
    ) -> list[str]:
        """Build complete Docker command with overrides.

        Args:
            docker_image: Override Docker image
            docker_env: Additional environment variables (KEY=VALUE format)
            docker_volume: Additional volume mounts (host:container[:options] format)
            docker_workdir: Override working directory
            docker_arg: Additional Docker run arguments
            docker_home: Override home directory inside container
            docker_workspace: Override workspace directory inside container
            user_mapping_enabled: Override user mapping enable/disable
            user_uid: Override user ID for container
            user_gid: Override group ID for container
            entrypoint: Override Docker entrypoint
            command: Command to run in container (list of strings)
            cmd_args: Arguments to pass to the command

        Returns:
            Complete Docker command as list of strings
        """
        # Start with base Docker command
        cmd = ["docker", "run", "--rm", "-it"]

        # Only create volumes if custom directories are specified
        volumes = []

        # Determine effective home and workspace directories
        home_dir = docker_home or self.settings.docker_home_directory
        workspace_dir = docker_workspace or self.settings.docker_workspace_directory

        # Validate and normalize overrides if provided
        if docker_home:
            home_dir = validate_host_path(docker_home)
        if docker_workspace:
            workspace_dir = validate_host_path(docker_workspace)

        # Create volumes based on directories only if they are specified
        if home_dir or workspace_dir:
            volumes = self._get_volumes_for_directories(home_dir, workspace_dir)

        # Add CLI volume overrides
        if docker_volume:
            normalized_volumes = []
            for volume in docker_volume:
                normalized_volume = validate_volume_format(volume)
                normalized_volumes.append(normalized_volume)
            volumes.extend(normalized_volumes)

        # Add default volumes from settings only if no custom directories were specified
        if not home_dir and not workspace_dir:
            volumes.extend(self.settings.docker_volumes)

        # Add volumes to command
        for volume in volumes:
            cmd.extend(["--volume", volume])

        # Create environment variables
        env_vars = self._get_environment_for_directories(home_dir, workspace_dir)

        # Add CLI environment overrides
        if docker_env:
            for env_var in docker_env:
                key, value = validate_environment_variable(env_var)
                env_vars[key] = value

        # Add environment variables to command
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Add user mapping if enabled
        effective_mapping_enabled = (
            user_mapping_enabled
            if user_mapping_enabled is not None
            else self.settings.user_mapping_enabled
        )

        if effective_mapping_enabled:
            effective_uid = user_uid if user_uid is not None else self.settings.user_uid
            effective_gid = user_gid if user_gid is not None else self.settings.user_gid

            if effective_uid is not None and effective_gid is not None:
                cmd.extend(["--user", f"{effective_uid}:{effective_gid}"])

        # Add working directory if specified
        if docker_workdir:
            cmd.extend(["--workdir", docker_workdir])

        # Add entrypoint if specified
        if entrypoint:
            cmd.extend(["--entrypoint", entrypoint])

        # Add additional Docker arguments
        additional_args = self._get_merged_additional_args(docker_arg)
        cmd.extend(additional_args)

        # Add Docker image
        image = docker_image or self.settings.docker_image
        cmd.append(image)

        # Add command and arguments if specified
        if command:
            cmd.extend(command)
            if cmd_args:
                cmd.extend(cmd_args)
        elif cmd_args:
            # If only arguments are provided, add them directly
            cmd.extend(cmd_args)

        return cmd

    def _get_merged_volumes(
        self, cli_volumes: list[str] | None, base_volumes: list[str] | None = None
    ) -> list[str]:
        """Merge configuration volumes with CLI overrides."""
        volumes = (base_volumes or self.settings.docker_volumes).copy()
        if cli_volumes:
            # Validate and normalize CLI volumes
            normalized_volumes = []
            for volume in cli_volumes:
                normalized_volume = validate_volume_format(volume)
                normalized_volumes.append(normalized_volume)
            volumes.extend(normalized_volumes)
        return volumes

    def _get_merged_environment(
        self, cli_env: list[str] | None, base_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Merge configuration environment with CLI overrides."""
        env = (base_env or self.settings.docker_environment).copy()

        if cli_env:
            for env_var in cli_env:
                key, value = validate_environment_variable(env_var)
                env[key] = value

        return env

    def _get_merged_additional_args(self, cli_args: list[str] | None) -> list[str]:
        """Merge configuration additional args with CLI overrides."""
        args = self.settings.docker_additional_args.copy()
        if cli_args:
            args.extend(cli_args)
        return args

    def _get_volumes_for_directories(
        self, home_dir: str | None, workspace_dir: str | None
    ) -> list[str]:
        """Create volume mounts for home and workspace directories.

        Args:
            home_dir: Host path for home directory (can be None)
            workspace_dir: Host path for workspace directory (can be None)
        """
        volumes = []
        if home_dir:
            volumes.append(f"{home_dir}:/data/home")
        if workspace_dir:
            volumes.append(f"{workspace_dir}:/data/workspace")
        return volumes

    def _get_environment_for_directories(
        self, home_dir: str | None, workspace_dir: str | None
    ) -> dict[str, str]:
        """Create environment variables for home and workspace directories.

        Args:
            home_dir: Host path for home directory (not used for env vars)
            workspace_dir: Host path for workspace directory (not used for env vars)
        """
        env = self.settings.docker_environment.copy()

        # Set the environment variables to point to container paths
        env["CLAUDE_HOME"] = "/data/home"
        env["CLAUDE_WORKSPACE"] = "/data/workspace"

        return env

    def execute(self, **kwargs: Any) -> None:
        """Build and execute Docker command using os.execvp.

        This method builds the Docker command and replaces the current process
        with the Docker command, effectively handing over control to Docker.

        Args:
            **kwargs: All arguments supported by build_command()

        Raises:
            OSError: If the command cannot be executed
        """
        cmd = self.build_command(**kwargs)
        logger.info("Executing Docker command: %s", cmd)
        os.execvp(cmd[0], cmd)

    def build_and_execute(self, **kwargs: Any) -> None:
        """Alias for execute method for backward compatibility."""
        self.execute(**kwargs)

    @classmethod
    def from_settings_and_overrides(
        cls,
        docker_settings: DockerSettings,
        **overrides: Any,
    ) -> list[str]:
        """Convenience method to build command directly.

        Args:
            docker_settings: Docker configuration from settings
            **overrides: CLI override arguments

        Returns:
            Complete Docker command as list of strings
        """
        builder = cls(docker_settings)
        return builder.build_command(**overrides)

    @classmethod
    def execute_from_settings(
        cls,
        docker_settings: DockerSettings,
        **overrides: Any,
    ) -> None:
        """Convenience method to build and execute command directly.

        Args:
            docker_settings: Docker configuration from settings
            **overrides: CLI override arguments

        Raises:
            OSError: If the command cannot be executed
        """
        builder = cls(docker_settings)
        builder.execute(**overrides)
