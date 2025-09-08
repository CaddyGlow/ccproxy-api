"""Docker adapter for executing commands in Docker containers."""

import os
import subprocess
from typing import Any

import structlog
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.streaming import DeferredStreaming

from .config import DockerConfig


logger = structlog.get_logger(__name__)


class DockerAdapter(BaseAdapter):
    """Adapter for Docker command execution."""

    def __init__(self, config: DockerConfig | None = None):
        """Initialize Docker adapter.

        Args:
            config: Docker configuration
        """
        self.config = config or DockerConfig()

    def build_docker_run_args(
        self,
        settings: Any,
        command: list[str] | None = None,
        docker_image: str | None = None,
        docker_env: list[str] | None = None,
        docker_volume: list[str] | None = None,
        docker_arg: list[str] | None = None,
        docker_home: str | None = None,
        docker_workspace: str | None = None,
        user_mapping_enabled: bool | None = None,
        user_uid: int | None = None,
        user_gid: int | None = None,
    ) -> tuple[str, list[str], list[str], list[str], dict[str, Any], dict[str, Any]]:
        """Build Docker run arguments.

        Returns:
            Tuple of (image, volumes, environment, command, user_context, metadata)
        """
        # Use CLI overrides or config defaults
        image = docker_image or self.config.docker_image
        home_dir = docker_home or str(self.config.get_effective_home_directory())
        workspace_dir = docker_workspace or str(
            self.config.get_effective_workspace_directory()
        )

        # Build volumes
        volumes = [
            f"{home_dir}:/data/home",
            f"{workspace_dir}:/data/workspace",
        ]
        volumes.extend(self.config.get_all_volumes(docker_volume))

        # Build environment variables
        env_vars = [
            "CLAUDE_HOME=/data/home",
            "CLAUDE_WORKSPACE=/data/workspace",
        ]
        env_vars.extend(self.config.get_all_environment_vars(docker_env))

        # User mapping
        user_context = {}
        if user_mapping_enabled is None:
            user_mapping_enabled = self.config.user_mapping_enabled

        if user_mapping_enabled:
            uid = user_uid or self.config.user_uid or os.getuid()
            gid = user_gid or self.config.user_gid or os.getgid()
            user_context = {"uid": uid, "gid": gid}

        metadata = {
            "config": self.config,
            "cli_overrides": {
                "docker_image": docker_image,
                "docker_env": docker_env,
                "docker_volume": docker_volume,
                "docker_arg": docker_arg,
                "docker_home": docker_home,
                "docker_workspace": docker_workspace,
                "user_mapping_enabled": user_mapping_enabled,
                "user_uid": user_uid,
                "user_gid": user_gid,
            },
        }

        return image, volumes, env_vars, command or [], user_context, metadata

    def exec_container(
        self,
        image: str,
        volumes: list[str],
        environment: list[str],
        command: list[str],
        user_context: dict[str, Any] | None = None,
        ports: list[str] | None = None,
    ) -> None:
        """Execute command in Docker container.

        Args:
            image: Docker image to use
            volumes: Volume mounts
            environment: Environment variables
            command: Command to execute
            user_context: User mapping context
            ports: Port mappings
        """
        docker_cmd = ["docker", "run", "--rm", "-i"]

        # Add user mapping if provided
        if user_context:
            uid = user_context.get("uid")
            gid = user_context.get("gid")
            if uid is not None and gid is not None:
                docker_cmd.extend(["--user", f"{uid}:{gid}"])

        # Add volumes
        for volume in volumes:
            docker_cmd.extend(["-v", volume])

        # Add environment variables
        for env_var in environment:
            docker_cmd.extend(["-e", env_var])

        # Add port mappings
        if ports:
            for port in ports:
                docker_cmd.extend(["-p", port])

        # Add image and command
        docker_cmd.append(image)
        if command:
            docker_cmd.extend(command)

        logger.info(
            "docker_execution",
            command=" ".join(docker_cmd),
            image=image,
            volumes_count=len(volumes),
            env_vars_count=len(environment),
            ports_count=len(ports) if ports else 0,
        )

        # Execute the Docker command
        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(
                "docker_execution_failed", error=str(e), exit_code=e.returncode
            )
            raise
        except FileNotFoundError:
            logger.error("docker_not_found", error="Docker not found in PATH")
            raise RuntimeError("Docker not found. Please install Docker.")

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request (not used for Docker adapter)."""
        raise NotImplementedError("Docker adapter does not handle HTTP requests")

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle streaming request (not used for Docker adapter)."""
        raise NotImplementedError("Docker adapter does not handle streaming requests")

    async def cleanup(self) -> None:
        """Cleanup Docker adapter resources."""
        # No persistent resources to cleanup for Docker adapter
        pass
