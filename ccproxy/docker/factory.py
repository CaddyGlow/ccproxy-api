"""Factory functions for creating Docker adapters with various configurations.

This module provides builder patterns and factory functions for creating
Docker adapters with common configurations.
"""

from pathlib import Path
from typing import Any

from .adapter import DockerAdapter, create_docker_adapter
from .docker_path import DockerPath
from .middleware import create_logger_middleware
from .models import DockerUserContext
from .protocol import DockerAdapterProtocol, DockerEnv, DockerVolume


def create_default_adapter(
    image: str = "ghcr.io/anthropics/claude-cli:latest",
    volumes: list[DockerVolume] | None = None,
    environment: DockerEnv | None = None,
) -> DockerAdapterProtocol:
    """Create a Docker adapter with default settings.

    Args:
        image: Docker image to use
        volumes: Optional list of volume mappings
        environment: Optional environment variables

    Returns:
        Configured Docker adapter
    """
    return create_docker_adapter(
        image=image,
        volumes=volumes or [],
        environment=environment or {},
    )


def create_development_adapter(
    image: str = "ghcr.io/anthropics/claude-cli:latest",
    project_dir: Path | None = None,
    enable_logging: bool = True,
) -> DockerAdapterProtocol:
    """Create a Docker adapter configured for development.

    Args:
        image: Docker image to use
        project_dir: Project directory to mount
        enable_logging: Whether to enable debug logging

    Returns:
        Configured Docker adapter for development
    """
    volumes: list[DockerVolume] = []
    environment: DockerEnv = {
        "CLAUDE_ENV": "development",
        "DEBUG": "1" if enable_logging else "0",
    }

    if project_dir:
        volumes.append((str(project_dir), "/workspace"))
        environment["CLAUDE_WORKSPACE"] = "/workspace"

    adapter = create_docker_adapter(
        image=image,
        volumes=volumes,
        environment=environment,
    )

    if enable_logging:
        # Add logging middleware for development
        original_run = adapter.run

        def run_with_logging(*args: Any, **kwargs: Any) -> Any:
            middleware = create_logger_middleware()
            if "output_middleware" in kwargs:
                # Chain with existing middleware
                from .middleware import create_chained_docker_middleware

                kwargs["output_middleware"] = create_chained_docker_middleware(
                    [middleware, kwargs["output_middleware"]]
                )
            else:
                kwargs["output_middleware"] = middleware
            return original_run(*args, **kwargs)

        adapter.run = run_with_logging  # type: ignore

    return adapter


def create_user_mapped_adapter(
    image: str = "ghcr.io/anthropics/claude-cli:latest",
    uid: int | None = None,
    gid: int | None = None,
    username: str | None = None,
    home_dir: Path | None = None,
    workspace_dir: Path | None = None,
) -> DockerAdapterProtocol:
    """Create a Docker adapter with user mapping.

    Args:
        image: Docker image to use
        uid: User ID for mapping
        gid: Group ID for mapping
        username: Username for the container
        home_dir: Home directory to mount
        workspace_dir: Workspace directory to mount

    Returns:
        Configured Docker adapter with user mapping
    """
    import getpass
    import os

    # Use current user's UID/GID if not provided
    if uid is None:
        uid = os.getuid()
    if gid is None:
        gid = os.getgid()
    if username is None:
        username = getpass.getuser()

    volumes: list[DockerVolume] = []
    environment: DockerEnv = {}

    # Create DockerPath instances for user context
    home_path = None
    workspace_path = None

    if home_dir:
        home_path = DockerPath(host_path=home_dir, container_path="/data/home")
        volumes.append((str(home_dir), "/data/home"))
        environment["CLAUDE_HOME"] = "/data/home"

    if workspace_dir:
        workspace_path = DockerPath(
            host_path=workspace_dir, container_path="/data/workspace"
        )
        volumes.append((str(workspace_dir), "/data/workspace"))
        environment["CLAUDE_WORKSPACE"] = "/data/workspace"

    user_context = DockerUserContext(
        uid=uid,
        gid=gid,
        username=username,
        home_path=home_path,
        workspace_path=workspace_path,
    )

    return create_docker_adapter(
        image=image,
        volumes=volumes,
        environment=environment,
        user_context=user_context,
    )


def create_production_adapter(
    image: str = "ghcr.io/anthropics/claude-cli:latest",
    volumes: list[DockerVolume] | None = None,
    environment: DockerEnv | None = None,
    resource_limits: dict[str, Any] | None = None,
) -> DockerAdapterProtocol:
    """Create a Docker adapter configured for production.

    Args:
        image: Docker image to use
        volumes: Optional list of volume mappings
        environment: Optional environment variables
        resource_limits: Optional resource limits (memory, cpu)

    Returns:
        Configured Docker adapter for production
    """
    prod_env: DockerEnv = {
        "CLAUDE_ENV": "production",
        "LOG_LEVEL": "WARNING",
    }
    if environment:
        prod_env.update(environment)

    additional_args = []
    if resource_limits:
        if "memory" in resource_limits:
            additional_args.extend(["--memory", resource_limits["memory"]])
        if "cpu" in resource_limits:
            additional_args.extend(["--cpus", str(resource_limits["cpu"])])

    return create_docker_adapter(
        image=image,
        volumes=volumes or [],
        environment=prod_env,
        additional_args=additional_args,
    )


def create_isolated_adapter(
    image: str = "ghcr.io/anthropics/claude-cli:latest",
    temp_workspace: bool = True,
) -> DockerAdapterProtocol:
    """Create an isolated Docker adapter with minimal host access.

    Args:
        image: Docker image to use
        temp_workspace: Whether to create a temporary workspace

    Returns:
        Configured Docker adapter with isolation
    """
    volumes: list[DockerVolume] = []
    environment: DockerEnv = {
        "CLAUDE_ENV": "isolated",
    }

    if temp_workspace:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="claude_"))
        volumes.append((str(temp_dir), "/workspace"))
        environment["CLAUDE_WORKSPACE"] = "/workspace"

    # Add security options for better isolation
    additional_args = [
        "--read-only",
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
    ]

    return create_docker_adapter(
        image=image,
        volumes=volumes,
        environment=environment,
        additional_args=additional_args,
    )
