"""Docker plugin configuration."""

from pathlib import Path

from pydantic import BaseModel, Field


class DockerConfig(BaseModel):
    """Configuration for Docker plugin."""

    enabled: bool = Field(
        default=True,
        description="Enable Docker functionality",
    )

    docker_image: str = Field(
        default="anthropics/claude-cli:latest",
        description="Docker image to use for running commands",
    )

    docker_home_directory: str | None = Field(
        default=None,
        description="Home directory to mount in Docker container",
    )

    docker_workspace_directory: str | None = Field(
        default=None,
        description="Workspace directory to mount in Docker container",
    )

    docker_volumes: list[str] = Field(
        default_factory=list,
        description="Additional volume mounts for Docker container",
    )

    docker_environment: list[str] = Field(
        default_factory=list,
        description="Environment variables to pass to Docker container",
    )

    user_mapping_enabled: bool = Field(
        default=True,
        description="Enable user mapping for Docker containers",
    )

    user_uid: int | None = Field(
        default=None,
        description="User UID for Docker user mapping",
    )

    user_gid: int | None = Field(
        default=None,
        description="User GID for Docker user mapping",
    )

    def get_effective_home_directory(self) -> Path:
        """Get the effective home directory for Docker mounting."""
        if self.docker_home_directory:
            return Path(self.docker_home_directory)
        return Path.home()

    def get_effective_workspace_directory(self) -> Path:
        """Get the effective workspace directory for Docker mounting."""
        if self.docker_workspace_directory:
            return Path(self.docker_workspace_directory)
        return Path.cwd()

    def get_all_volumes(self, additional_volumes: list[str] | None = None) -> list[str]:
        """Get all volume mounts including defaults and additional."""
        volumes = self.docker_volumes.copy()
        if additional_volumes:
            volumes.extend(additional_volumes)
        return volumes

    def get_all_environment_vars(
        self, additional_env: list[str] | None = None
    ) -> list[str]:
        """Get all environment variables including defaults and additional."""
        env_vars = self.docker_environment.copy()
        if additional_env:
            env_vars.extend(additional_env)
        return env_vars
