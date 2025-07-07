"""Docker settings configuration for Claude Code Proxy API."""

import os
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from claude_code_proxy import __version__
from claude_code_proxy.utils.docker_validation import (
    validate_docker_volumes as validate_volumes_list,
)
from claude_code_proxy.utils.docker_validation import validate_host_path
from claude_code_proxy.utils.version import format_version
from claude_code_proxy.utils.xdg import get_claude_docker_home_dir


class DockerSettings(BaseModel):
    """Docker configuration settings for running Claude commands in containers."""

    docker_image: str = Field(
        default=f"ghcr.io/caddyglow/claude-code-proxy-api:{format_version(__version__, 'docker')}",
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
