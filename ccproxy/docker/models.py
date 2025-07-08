"""Docker-specific models for cross-domain operations."""

import os
import platform
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from .docker_path import DockerPath


class DockerUserContext(BaseModel):
    """Docker user context for volume permission handling.

    Represents user information needed for Docker --user flag to
    solve volume permission issues when mounting host directories.
    """

    uid: int = Field(..., description="User ID for Docker --user flag")
    gid: int = Field(..., description="Group ID for Docker --user flag")
    username: str = Field(..., description="Username for reference")
    enable_user_mapping: bool = Field(
        default=True, description="Whether to enable --user flag in Docker commands"
    )

    # Path settings using DockerPath
    home_path: DockerPath | None = Field(
        default=None, description="Home directory mapping between host and container"
    )
    workspace_path: DockerPath | None = Field(
        default=None,
        description="Workspace directory mapping between host and container",
    )

    # Platform compatibility
    _supported_platforms: ClassVar[set[str]] = {"Linux", "Darwin"}

    @field_validator("uid", "gid")
    @classmethod
    def validate_positive_ids(cls, v: int) -> int:
        """Validate that UID/GID are positive integers."""
        if v < 0:
            raise ValueError("UID and GID must be non-negative")
        return v

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username is not empty."""
        if not v or not v.strip():
            raise ValueError("Username cannot be empty")
        return v.strip()

    @classmethod
    def detect_current_user(
        cls,
        home_path: DockerPath | None = None,
        workspace_path: DockerPath | None = None,
    ) -> "DockerUserContext":
        """Detect current user context from system.

        Args:
            home_path: Optional home directory DockerPath override
            workspace_path: Optional workspace directory DockerPath override

        Returns:
            DockerUserContext: Current user's context

        Raises:
            RuntimeError: If user detection fails or platform unsupported
        """
        current_platform = platform.system()

        if current_platform not in cls._supported_platforms:
            raise RuntimeError(
                f"User detection not supported on {current_platform}. "
                f"Supported platforms: {', '.join(cls._supported_platforms)}"
            )

        try:
            uid = os.getuid()
            gid = os.getgid()
            username = os.getenv("USER") or os.getenv("USERNAME") or "unknown"

            # Create default home path if not provided
            if home_path is None:
                host_home_env = os.getenv("HOME")
                if host_home_env:
                    home_path = DockerPath(
                        host_path=Path(host_home_env), container_path="/data/home"
                    )

            return cls(
                uid=uid,
                gid=gid,
                username=username,
                enable_user_mapping=True,
                home_path=home_path,
                workspace_path=workspace_path,
            )

        except AttributeError as e:
            raise RuntimeError(
                f"Failed to detect user on {current_platform}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error detecting user: {e}") from e

    @classmethod
    def create_manual(
        cls,
        uid: int,
        gid: int,
        username: str,
        home_path: DockerPath | None = None,
        workspace_path: DockerPath | None = None,
        enable_user_mapping: bool = True,
    ) -> "DockerUserContext":
        """Create manual user context with custom values.

        Args:
            uid: User ID for Docker --user flag
            gid: Group ID for Docker --user flag
            username: Username for reference
            home_path: Optional home directory DockerPath
            workspace_path: Optional workspace directory DockerPath
            enable_user_mapping: Whether to enable --user flag in Docker commands

        Returns:
            DockerUserContext: Manual user context

        Raises:
            ValueError: If validation fails for any parameter
        """
        return cls(
            uid=uid,
            gid=gid,
            username=username,
            enable_user_mapping=enable_user_mapping,
            home_path=home_path,
            workspace_path=workspace_path,
        )

    def get_docker_user_flag(self) -> str:
        """Get Docker --user flag value.

        Returns:
            str: Docker user flag in format "uid:gid"
        """
        return f"{self.uid}:{self.gid}"

    def is_supported_platform(self) -> bool:
        """Check if current platform supports user mapping.

        Returns:
            bool: True if platform supports user mapping
        """
        return platform.system() in self._supported_platforms

    def should_use_user_mapping(self) -> bool:
        """Check if user mapping should be used.

        Returns:
            bool: True if user mapping is enabled and platform is supported
        """
        return self.enable_user_mapping and self.is_supported_platform()

    def get_environment_variables(self) -> dict[str, str]:
        """Get environment variables for home and workspace directory configuration.

        Returns:
            dict[str, str]: Environment variables to set in container
        """
        env = {}
        if self.home_path:
            env["HOME"] = self.home_path.container()
            env["CLAUDE_HOME"] = self.home_path.container()
        if self.workspace_path:
            env["CLAUDE_WORKSPACE"] = self.workspace_path.container()
        return env

    def get_volumes(self) -> list[tuple[str, str]]:
        """Get Docker volume mappings for home and workspace directories.

        Returns:
            list[tuple[str, str]]: List of (host_path, container_path) tuples
        """
        volumes = []
        if self.home_path and self.home_path.host_path:
            volumes.append(self.home_path.vol())
        if self.workspace_path and self.workspace_path.host_path:
            volumes.append(self.workspace_path.vol())
        return volumes

    def get_home_volumes(self) -> list[tuple[str, str]]:
        """Get Docker volume mappings for home directory only (for backwards compatibility).

        Returns:
            list[tuple[str, str]]: List of (host_path, container_path) tuples
        """
        volumes = []
        if self.home_path and self.home_path.host_path:
            volumes.append(self.home_path.vol())
        return volumes

    def describe_context(self) -> str:
        """Get human-readable description of user context.

        Returns:
            str: Description of user context for debugging
        """
        parts = [
            f"uid={self.uid}",
            f"gid={self.gid}",
            f"username={self.username}",
        ]

        if self.home_path:
            parts.append(f"home_path={self.home_path}")

        if self.workspace_path:
            parts.append(f"workspace_path={self.workspace_path}")

        return f"DockerUserContext({', '.join(parts)})"


__all__ = ["DockerUserContext"]
