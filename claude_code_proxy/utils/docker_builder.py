"""Docker command builder utility for Claude Proxy."""

from pathlib import Path
from typing import Any

from ..config.settings import DockerSettings


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
            home_dir = self._validate_host_path(docker_home)
        if docker_workspace:
            workspace_dir = self._validate_host_path(docker_workspace)

        # Create volumes based on directories only if they are specified
        if home_dir or workspace_dir:
            volumes = self._get_volumes_for_directories(home_dir, workspace_dir)

        # Add CLI volume overrides
        if docker_volume:
            normalized_volumes = []
            for volume in docker_volume:
                normalized_volume = self._validate_volume_format(volume)
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
                if "=" not in env_var:
                    raise ValueError(
                        f"Invalid environment variable format: '{env_var}'. Expected KEY=VALUE"
                    )
                key, value = env_var.split("=", 1)
                env_vars[key] = value

        # Add environment variables to command
        for key, value in env_vars.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Add additional Docker arguments
        additional_args = self._get_merged_additional_args(docker_arg)
        cmd.extend(additional_args)

        # Add Docker image
        image = docker_image or self.settings.docker_image
        cmd.append(image)

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
                normalized_volume = self._validate_volume_format(volume)
                normalized_volumes.append(normalized_volume)
            volumes.extend(normalized_volumes)
        return volumes

    def _validate_volume_format(self, volume: str) -> str:
        """Validate and normalize volume mount format.

        Returns:
            Normalized volume string with absolute host path
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

        # Convert relative paths to absolute
        host_path = parts[0]
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

    def _validate_working_directory(self, workdir: str) -> str:
        """Validate and normalize Docker working directory format.

        Returns:
            Normalized working directory as absolute path
        """
        path_obj = Path(workdir)

        # If it's a relative path, convert to absolute
        if not path_obj.is_absolute():
            workdir = str(path_obj.resolve())

        return workdir

    def _validate_host_path(self, path: str) -> str:
        """Validate and normalize host path format.

        Returns:
            Normalized host path as absolute path
        """
        import os

        # Expand environment variables
        expanded_path = os.path.expandvars(path)
        path_obj = Path(expanded_path)

        # If it's a relative path, convert to absolute
        if not path_obj.is_absolute():
            path = str(path_obj.resolve())
        else:
            path = expanded_path

        return path

    def _get_merged_environment(
        self, cli_env: list[str] | None, base_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Merge configuration environment with CLI overrides."""
        env = (base_env or self.settings.docker_environment).copy()

        if cli_env:
            for env_var in cli_env:
                if "=" not in env_var:
                    raise ValueError(
                        f"Invalid environment variable format: '{env_var}'. Expected KEY=VALUE"
                    )
                key, value = env_var.split("=", 1)
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
