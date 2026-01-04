"""Docker path management with clean API."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, field_validator

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger(__name__)


class DockerPath(BaseModel):
    """Represents a mapping between host and container paths.

    Provides a clean API for Docker volume mounting and path resolution.

    Example:
        workspace = DockerPath(host_path="/some/host/local/path", container_path="/tmp/docker/workspace")
        docker_vol = workspace.vol()  # Returns volume mapping tuple
        container_path = workspace.container()  # Returns container path
        host_path = workspace.host()  # Returns host path
    """

    host_path: Path | None = None
    container_path: str
    env_definition_variable_name: str | None = None

    @field_validator("host_path", mode="before")
    @classmethod
    def _resolve_host_path(cls, v: str | Path | None) -> Path | None:
        """Resolve host path to an absolute path."""
        if v is None:
            return None
        return Path(v).resolve()

    def vol(self) -> tuple[str, str]:
        """Get Docker volume mapping tuple.

        Returns:
            tuple[str, str]: (host_path, container_path) for Docker -v flag
        """
        if self.host_path is None:
            raise ValueError("host_path is not set, cannot create volume mapping")
        return (str(self.host_path), self.container_path)

    def host(self) -> Path:
        """Get host path as Path object.

        Returns:
            Path: Resolved host path
        """
        if self.host_path is None:
            raise ValueError("host_path is not set")
        return self.host_path

    def container(self) -> str:
        """Get container path as string.

        Returns:
            str: Container path
        """
        return self.container_path

    def join(self, *subpaths: str) -> "DockerPath":
        """Create new DockerPath with subpaths joined to both host and container paths.

        Args:
            *subpaths: Path components to join

        Returns:
            DockerPath: New instance with joined paths
        """
        host_joined = self.host_path
        if host_joined:
            for subpath in subpaths:
                host_joined = host_joined / subpath

        container_joined = self.container_path
        for subpath in subpaths:
            container_joined = f"{container_joined}/{subpath}".replace("//", "/")

        return DockerPath(host_path=host_joined, container_path=container_joined)

    def get_env_definition(self) -> str:
        return f"{self.env_definition_variable_name}={self.container_path} # {self.host_path}"

    def __str__(self) -> str:
        """String representation showing the mapping."""
        if self.host_path:
            return f"DockerPath({self.host_path} -> {self.container_path})"
        return f"DockerPath(container_path={self.container_path})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"DockerPath(host_path={self.host_path!r}, container_path={self.container_path!r})"


class DockerPathSet:
    """Collection of named Docker paths for organized path management.

    Example:
        paths = DockerPathSet("/tmp/build")
        paths.add("workspace", "/workspace")
        paths.add("config", "/workspace/config")

        workspace_vol = paths.get("workspace").vol()
        config_path = paths.get("config").container()
    """

    def __init__(self, base_host_path: str | Path | None = None) -> None:
        """Initialize Docker path set.

        Args:
            base_host_path: Base path on host for all paths in this set
        """
        self.base_host_path = Path(base_host_path).resolve() if base_host_path else None
        self.paths: dict[str, DockerPath] = {}
        self.logger = get_plugin_logger(f"{__name__}.{self.__class__.__name__}")

    def add(
        self, name: str, container_path: str, host_subpath: str | None = None
    ) -> Self:
        """Add a named Docker path to the set.

        Args:
            name: Logical name for the path
            container_path: Path inside the Docker container
            host_subpath: Optional subpath from base_host_path, defaults to name

        Returns:
            Self: For method chaining
        """
        if self.base_host_path is None:
            raise ValueError("base_host_path must be set to use add() method")

        if host_subpath is None:
            host_subpath = name

        # Handle empty string to mean no subpath (use base path directly)
        if host_subpath == "":
            host_path = self.base_host_path
        else:
            host_path = self.base_host_path / host_subpath

        self.paths[name] = DockerPath(
            host_path=host_path, container_path=container_path
        )
        return self

    def add_path(self, name: str, docker_path: DockerPath) -> Self:
        """Add a pre-created DockerPath to the set.

        Args:
            name: Logical name for the path
            docker_path: DockerPath instance to add

        Returns:
            Self: For method chaining
        """
        self.paths[name] = docker_path
        return self

    def get(self, name: str) -> DockerPath:
        """Get Docker path by name.

        Args:
            name: Logical name of the path

        Returns:
            DockerPath: The Docker path instance

        Raises:
            KeyError: If path name is not found
        """
        if name not in self.paths:
            raise KeyError(
                f"Docker path '{name}' not found. Available: {list(self.paths.keys())}"
            )
        return self.paths[name]

    def has(self, name: str) -> bool:
        """Check if a path name exists in the set.

        Args:
            name: Logical name to check

        Returns:
            bool: True if path exists
        """
        return name in self.paths

    def volumes(self) -> list[tuple[str, str]]:
        """Get all volume mappings for Docker.

        Returns:
            list[tuple[str, str]]: List of (host_path, container_path) tuples
        """
        return [path.vol() for path in self.paths.values()]

    def names(self) -> list[str]:
        """Get all path names in the set.

        Returns:
            list[str]: List of logical path names
        """
        return list(self.paths.keys())
