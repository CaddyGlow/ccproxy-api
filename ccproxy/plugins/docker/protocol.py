"""Protocol definition for Docker operations."""

from collections.abc import Awaitable
from pathlib import Path
from typing import (
    Protocol,
    TypeAlias,
    runtime_checkable,
)

from .models import DockerUserContext
from .stream_process import OutputMiddleware, ProcessResult, T


# Type aliases for Docker operations
DockerVolume: TypeAlias = tuple[str, str]  # (host_path, container_path)
DockerEnv: TypeAlias = dict[str, str]  # Environment variables
DockerPortSpec: TypeAlias = str  # Port specification (e.g., "8080:80", "localhost:8080:80", "127.0.0.1:8080:80/tcp")
DockerResult: TypeAlias = tuple[
    int, list[str], list[str]
]  # (return_code, stdout, stderr)


# TODO: add get_version, image_info,
@runtime_checkable
class DockerAdapterProtocol(Protocol):
    """Protocol for Docker operations."""

    def is_available(self) -> Awaitable[bool]:
        """Check if Docker is available on the system.

        Returns:
            True if Docker is available, False otherwise
        """
        ...

    def run(
        self,
        image: str,
        volumes: list[DockerVolume],
        environment: DockerEnv,
        command: list[str] | None = None,
        middleware: OutputMiddleware[T] | None = None,
        user_context: DockerUserContext | None = None,
        entrypoint: str | None = None,
        ports: list[DockerPortSpec] | None = None,
    ) -> Awaitable[ProcessResult[T]]:
        """Run a Docker container with specified configuration.

        Alias for run_container method.

        Args:
            image: Docker image name/tag to run
            volumes: List of volume mounts (host_path, container_path)
            environment: Dictionary of environment variables
            command: Optional command to run in the container
            middleware: Optional middleware for processing output
            user_context: Optional user context for Docker --user flag
            entrypoint: Optional custom entrypoint to override the image's default
            ports: Optional port specifications (e.g., ["8080:80", "localhost:9000:9000"])

        Returns:
            Tuple containing (return_code, stdout_lines, stderr_lines)

        Raises:
            DockerError: If the container fails to run
        """
        ...

    def run_container(
        self,
        image: str,
        volumes: list[DockerVolume],
        environment: DockerEnv,
        command: list[str] | None = None,
        middleware: OutputMiddleware[T] | None = None,
        user_context: DockerUserContext | None = None,
        entrypoint: str | None = None,
        ports: list[DockerPortSpec] | None = None,
    ) -> Awaitable[ProcessResult[T]]:
        """Run a Docker container with specified configuration.

        Args:
            image: Docker image name/tag to run
            volumes: List of volume mounts (host_path, container_path)
            environment: Dictionary of environment variables
            command: Optional command to run in the container
            middleware: Optional middleware for processing output
            user_context: Optional user context for Docker --user flag
            entrypoint: Optional custom entrypoint to override the image's default
            ports: Optional port specifications (e.g., ["8080:80", "localhost:9000:9000"])

        Returns:
            Tuple containing (return_code, stdout_lines, stderr_lines)

        Raises:
            DockerError: If the container fails to run
        """
        ...

    def exec_container(
        self,
        image: str,
        volumes: list[DockerVolume],
        environment: DockerEnv,
        command: list[str] | None = None,
        user_context: DockerUserContext | None = None,
        entrypoint: str | None = None,
        ports: list[DockerPortSpec] | None = None,
    ) -> None:
        """Execute a Docker container by replacing the current process.

        This method builds the Docker command and replaces the current process
        with the Docker command using os.execvp, effectively handing over control to Docker.

        Args:
            image: Docker image name/tag to run
            volumes: List of volume mounts (host_path, container_path)
            environment: Dictionary of environment variables
            command: Optional command to run in the container
            user_context: Optional user context for Docker --user flag
            entrypoint: Optional custom entrypoint to override the image's default
            ports: Optional port specifications (e.g., ["8080:80", "localhost:9000:9000"])

        Raises:
            DockerError: If the container fails to execute
            OSError: If the command cannot be executed
        """
        ...

    def build_image(
        self,
        dockerfile_dir: Path,
        image_name: str,
        image_tag: str = "latest",
        no_cache: bool = False,
        middleware: OutputMiddleware[T] | None = None,
    ) -> Awaitable[ProcessResult[T]]:
        """Build a Docker image from a Dockerfile.

        Args:
            dockerfile_dir: Directory containing the Dockerfile
            image_name: Name to tag the built image with
            image_tag: Tag to use for the image
            no_cache: Whether to use Docker's cache during build
            middleware: Optional middleware for processing output

        Returns:
            ProcessResult containing (return_code, stdout_lines, stderr_lines)

        Raises:
            DockerError: If the image fails to build
        """
        ...

    def image_exists(
        self, image_name: str, image_tag: str = "latest"
    ) -> Awaitable[bool]:
        """Check if a Docker image exists locally.

        Args:
            image_name: Name of the image to check
            image_tag: Tag of the image to check

        Returns:
            True if the image exists locally, False otherwise
        """
        ...

    def pull_image(
        self,
        image_name: str,
        image_tag: str = "latest",
        middleware: OutputMiddleware[T] | None = None,
    ) -> Awaitable[ProcessResult[T]]:
        """Pull a Docker image from registry.

        Args:
            image_name: Name of the image to pull
            image_tag: Tag of the image to pull
            middleware: Optional middleware for processing output

        Returns:
            ProcessResult containing (return_code, stdout_lines, stderr_lines)

        Raises:
            DockerError: If the image fails to pull
        """
        ...
