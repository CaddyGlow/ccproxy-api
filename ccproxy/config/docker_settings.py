"""Docker settings configuration for CCProxy API."""

import os

from pydantic import BaseModel, Field, field_validator, model_validator

from ccproxy import __version__
from ccproxy.core.async_utils import format_version, get_claude_docker_home_dir


# Docker validation functions moved here to avoid utils dependency


def validate_host_path(path: str) -> str:
    """Validate host path for Docker volume mounting."""
    import os
    from pathlib import Path

    if not path:
        raise ValueError("Path cannot be empty")

    # Expand environment variables and user home directory
    expanded_path = os.path.expandvars(str(Path(path).expanduser()))

    # Convert to absolute path and normalize
    abs_path = Path(expanded_path).resolve()
    return str(abs_path)


def validate_volumes_list(volumes: list[str]) -> list[str]:
    """Validate Docker volumes list format."""
    validated = []

    for volume in volumes:
        if not volume:
            continue

        # Use validate_volume_format for comprehensive validation
        validated_volume = validate_volume_format(volume)
        validated.append(validated_volume)

    return validated


def validate_volume_format(volume: str) -> str:
    """Validate individual Docker volume format.

    Args:
        volume: Volume mount string in format 'host:container[:options]'

    Returns:
        Validated volume string with normalized host path

    Raises:
        ValueError: If volume format is invalid or host path doesn't exist
    """
    import os
    from pathlib import Path

    if not volume:
        raise ValueError("Volume cannot be empty")

    # Expected format: "host_path:container_path" or "host_path:container_path:options"
    parts = volume.split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid volume format: {volume}. Expected 'host:container' or 'host:container:options'"
        )

    host_path = parts[0]
    container_path = parts[1]
    options = ":".join(parts[2:]) if len(parts) > 2 else ""

    if not host_path or not container_path:
        raise ValueError(
            f"Invalid volume format: {volume}. Expected 'host:container' or 'host:container:options'"
        )

    # Expand environment variables and user home directory
    expanded_host_path = os.path.expandvars(str(Path(host_path).expanduser()))

    # Convert to absolute path
    abs_host_path = Path(expanded_host_path).resolve()

    # Check if the path exists
    if not abs_host_path.exists():
        raise ValueError(f"Host path does not exist: {expanded_host_path}")

    # Validate container path (should be absolute)
    if not container_path.startswith("/"):
        raise ValueError(f"Container path must be absolute: {container_path}")

    # Reconstruct the volume string with normalized host path
    result = f"{abs_host_path}:{container_path}"
    if options:
        result += f":{options}"

    return result


def validate_environment_variable(env_var: str) -> tuple[str, str]:
    """Validate environment variable format.

    Args:
        env_var: Environment variable string in format 'KEY=VALUE'

    Returns:
        Tuple of (key, value)

    Raises:
        ValueError: If environment variable format is invalid
    """
    if not env_var:
        raise ValueError("Environment variable cannot be empty")

    if "=" not in env_var:
        raise ValueError(
            f"Invalid environment variable format: {env_var}. Expected KEY=VALUE format"
        )

    # Split on first equals sign only (value may contain equals)
    key, value = env_var.split("=", 1)

    if not key:
        raise ValueError(
            f"Invalid environment variable format: {env_var}. Expected KEY=VALUE format"
        )

    return key, value


def validate_docker_volumes(volumes: list[str]) -> list[str]:
    """Validate Docker volumes list format.

    Args:
        volumes: List of volume mount strings

    Returns:
        List of validated volume strings with normalized host paths

    Raises:
        ValueError: If any volume format is invalid
    """
    validated = []

    for volume in volumes:
        if not volume:
            continue

        validated_volume = validate_volume_format(volume)
        validated.append(validated_volume)

    return validated


class DockerSettings(BaseModel):
    """Docker configuration settings for running Claude commands in containers."""

    docker_image: str = Field(
        default=f"ghcr.io/caddyglow/ccproxy-api:{format_version(__version__, level='docker')}",
        description="Docker image to use for Claude commands",
    )

    docker_volumes: list[str] = Field(
        default_factory=list,
        description="List of volume mounts in 'host:container[:options]' format",
    )

    docker_environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to pass to Docker container",
    )

    docker_additional_args: list[str] = Field(
        default_factory=list,
        description="Additional arguments to pass to docker run command",
    )

    docker_home_directory: str | None = Field(
        default=None,
        description="Local host directory to mount as the home directory in container",
    )

    docker_workspace_directory: str | None = Field(
        default=None,
        description="Local host directory to mount as the workspace directory in container",
    )

    user_mapping_enabled: bool = Field(
        default=True,
        description="Enable/disable UID/GID mapping for container user",
    )

    user_uid: int | None = Field(
        default=None,
        description="User ID to run container as (auto-detect current user if None)",
        ge=0,
    )

    user_gid: int | None = Field(
        default=None,
        description="Group ID to run container as (auto-detect current user if None)",
        ge=0,
    )

    @field_validator("docker_volumes")
    @classmethod
    def validate_docker_volumes(cls, v: list[str]) -> list[str]:
        """Validate Docker volume mount format."""
        return validate_volumes_list(v)

    @field_validator("docker_home_directory")
    @classmethod
    def validate_docker_home_directory(cls, v: str | None) -> str | None:
        """Validate and normalize Docker home directory (host path)."""
        if v is None:
            return None
        return validate_host_path(v)

    @field_validator("docker_workspace_directory")
    @classmethod
    def validate_docker_workspace_directory(cls, v: str | None) -> str | None:
        """Validate and normalize Docker workspace directory (host path)."""
        if v is None:
            return None
        return validate_host_path(v)

    @model_validator(mode="after")
    def setup_docker_configuration(self) -> "DockerSettings":
        """Set up Docker volumes and user mapping configuration."""
        # Set up Docker volumes based on home and workspace directories
        if (
            not self.docker_volumes
            and not self.docker_home_directory
            and not self.docker_workspace_directory
        ):
            # Use XDG config directory for Claude CLI data
            claude_config_dir = get_claude_docker_home_dir()
            home_host_path = str(claude_config_dir)
            workspace_host_path = os.path.expandvars("$PWD")

            self.docker_volumes = [
                f"{home_host_path}:/data/home",
                f"{workspace_host_path}:/data/workspace",
            ]

        # Update environment variables to point to container paths
        if "CLAUDE_HOME" not in self.docker_environment:
            self.docker_environment["CLAUDE_HOME"] = "/data/home"
        if "CLAUDE_WORKSPACE" not in self.docker_environment:
            self.docker_environment["CLAUDE_WORKSPACE"] = "/data/workspace"

        # Set up user mapping with auto-detection if enabled but not configured
        if self.user_mapping_enabled and os.name == "posix":
            # Auto-detect current user UID/GID if not explicitly set
            if self.user_uid is None:
                self.user_uid = os.getuid()
            if self.user_gid is None:
                self.user_gid = os.getgid()
        elif self.user_mapping_enabled and os.name != "posix":
            # Disable user mapping on non-Unix systems (Windows)
            self.user_mapping_enabled = False

        return self
