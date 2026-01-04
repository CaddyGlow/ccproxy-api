"""Docker adapter for container operations."""

import asyncio
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, cast

from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.streaming import DeferredStreaming

from .config import DockerConfig
from .middleware import LoggerOutputMiddleware
from .models import DockerUserContext
from .protocol import (
    DockerAdapterProtocol,
    DockerEnv,
    DockerPortSpec,
    DockerVolume,
)
from .stream_process import (
    OutputMiddleware,
    ProcessResult,
    T,
    run_command,
)
from .validators import create_docker_error, validate_port_spec


logger = get_plugin_logger(__name__)


class DockerAdapter(BaseAdapter, DockerAdapterProtocol):
    """Docker adapter implementing both BaseAdapter and DockerAdapterProtocol."""

    def __init__(self, config: DockerConfig | None = None):
        """Initialize Docker adapter.

        Args:
            config: Docker configuration
        """
        self.config = config or DockerConfig()

    async def _needs_sudo(self) -> bool:
        """Check if Docker requires sudo by testing docker info command."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode == 0:
                return False
            # Check if error suggests permission issues
            stderr_text = stderr.decode() if stderr else ""
            return (
                "permission denied" in stderr_text.lower()
                or "dial unix" in stderr_text.lower()
                or "connect: permission denied" in stderr_text.lower()
            )
        except Exception:
            return False

    async def is_available(self) -> bool:
        """Check if Docker is available on the system."""
        docker_cmd = ["docker", "--version"]
        cmd_str = " ".join(docker_cmd)

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                docker_version = stdout.decode().strip()
                logger.debug("docker_available", version=docker_version)
                return True
            else:
                stderr_text = stderr.decode() if stderr else "unknown error"
                logger.warning(
                    "docker_command_failed", command=cmd_str, error=stderr_text
                )
                return False

        except FileNotFoundError:
            logger.warning("docker_executable_not_found")
            return False

        except Exception as e:
            logger.warning("docker_availability_check_error", error=str(e))
            return False

    async def _run_with_sudo_fallback(
        self, docker_cmd: list[str], middleware: OutputMiddleware[T]
    ) -> ProcessResult[T]:
        # Try without sudo first
        try:
            result = await run_command(docker_cmd, middleware)
            return result
        except Exception as e:
            # Check if this might be a permission error
            error_text = str(e).lower()
            if any(
                phrase in error_text
                for phrase in [
                    "permission denied",
                    "dial unix",
                    "connect: permission denied",
                ]
            ):
                logger.info("docker_permission_denied_using_sudo")
                sudo_cmd = ["sudo"] + docker_cmd
                return await run_command(sudo_cmd, middleware)
            # Re-raise if not a permission error
            raise

    async def run_container(
        self,
        image: str,
        volumes: list[DockerVolume],
        environment: DockerEnv,
        command: list[str] | None = None,
        middleware: OutputMiddleware[T] | None = None,
        user_context: DockerUserContext | None = None,
        entrypoint: str | None = None,
        ports: list[DockerPortSpec] | None = None,
    ) -> ProcessResult[T]:
        """Run a Docker container with specified configuration."""

        docker_cmd = ["docker", "run", "--rm"]

        # Add user context if provided and should be used
        if user_context and user_context.should_use_user_mapping():
            docker_user_flag = user_context.get_docker_user_flag()
            docker_cmd.extend(["--user", docker_user_flag])
            logger.debug("docker_user_mapping", user_flag=docker_user_flag)

        # Add custom entrypoint if specified
        if entrypoint:
            docker_cmd.extend(["--entrypoint", entrypoint])
            logger.debug("docker_custom_entrypoint", entrypoint=entrypoint)

        # Add port publishing if specified
        if ports:
            for port_spec in ports:
                validated_port = validate_port_spec(port_spec)
                docker_cmd.extend(["-p", validated_port])
                logger.debug("docker_port_mapping", port=validated_port)

        # Add volume mounts
        for host_path, container_path in volumes:
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add environment variables
        for key, value in environment.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image
        docker_cmd.append(image)

        # Add command if specified
        if command:
            docker_cmd.extend(command)

        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)
        logger.debug("docker_command", command=cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            # Try with sudo fallback if needed
            result = await self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("docker_executable_not_found", error=str(e))
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Failed to run Docker container: {e}",
                cmd_str,
                e,
                {
                    "image": image,
                    "volumes_count": len(volumes),
                    "env_vars_count": len(environment),
                },
            )
            logger.error("docker_container_run_error", error=str(e))
            raise error from e

    async def run(
        self,
        image: str,
        volumes: list[DockerVolume],
        environment: DockerEnv,
        command: list[str] | None = None,
        middleware: OutputMiddleware[T] | None = None,
        user_context: DockerUserContext | None = None,
        entrypoint: str | None = None,
        ports: list[DockerPortSpec] | None = None,
    ) -> ProcessResult[T]:
        """Run a Docker container with specified configuration.

        This is an alias for run_container method.
        """
        return await self.run_container(
            image=image,
            volumes=volumes,
            environment=environment,
            command=command,
            middleware=middleware,
            user_context=user_context,
            entrypoint=entrypoint,
            ports=ports,
        )

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
        docker_cmd = ["docker", "run", "--rm", "-it"]

        # Add user context if provided and should be used
        if user_context and user_context.should_use_user_mapping():
            docker_user_flag = user_context.get_docker_user_flag()
            docker_cmd.extend(["--user", docker_user_flag])
            logger.debug("docker_user_mapping", user_flag=docker_user_flag)

        # Add custom entrypoint if specified
        if entrypoint:
            docker_cmd.extend(["--entrypoint", entrypoint])
            logger.debug("docker_custom_entrypoint", entrypoint=entrypoint)

        # Add port publishing if specified
        if ports:
            for port_spec in ports:
                validated_port = validate_port_spec(port_spec)
                docker_cmd.extend(["-p", validated_port])
                logger.debug("docker_port_mapping", port=validated_port)

        # Add volume mounts
        for host_path, container_path in volumes:
            docker_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Add environment variables
        for key, value in environment.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add image
        docker_cmd.append(image)

        # Add command if specified
        if command:
            docker_cmd.extend(command)

        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)
        logger.info("docker_execvp", command=cmd_str)

        try:
            # Check if we need sudo (without running the actual command)
            # Note: We can't use await here since this method replaces the process
            # Use a simple check instead
            try:
                subprocess.run(
                    ["docker", "info"], check=True, capture_output=True, text=True
                )
                needs_sudo = False
            except subprocess.CalledProcessError as e:
                needs_sudo = e.stderr and (
                    "permission denied" in e.stderr.lower()
                    or "dial unix" in e.stderr.lower()
                    or "connect: permission denied" in e.stderr.lower()
                )
            except Exception:
                needs_sudo = False

            if needs_sudo:
                logger.info("docker_using_sudo_for_execution")
                docker_cmd = ["sudo"] + docker_cmd

            # Replace current process with Docker command
            os.execvp(docker_cmd[0], docker_cmd)

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("docker_execvp_executable_not_found", error=str(e))
            raise error from e

        except OSError as e:
            error = create_docker_error(
                f"Failed to execute Docker command: {e}", cmd_str, e
            )
            logger.error("docker_execvp_os_error", error=str(e))
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Unexpected error executing Docker container: {e}",
                cmd_str,
                e,
                {
                    "image": image,
                    "volumes_count": len(volumes),
                    "env_vars_count": len(environment),
                },
            )
            logger.error("docker_execvp_unexpected_error", error=str(e))
            raise error from e

    async def build_image(
        self,
        dockerfile_dir: Path,
        image_name: str,
        image_tag: str = "latest",
        no_cache: bool = False,
        middleware: OutputMiddleware[T] | None = None,
    ) -> ProcessResult[T]:
        """Build a Docker image from a Dockerfile."""

        image_full_name = f"{image_name}:{image_tag}"

        # Check Docker availability
        if not await self.is_available():
            error = create_docker_error(
                "Docker is not available or not properly installed",
                None,
                None,
                {"image": image_full_name},
            )
            logger.error("docker_not_available_for_build", image=image_full_name)
            raise error

        # Validate dockerfile directory
        dockerfile_dir = Path(dockerfile_dir).resolve()
        if not dockerfile_dir.exists() or not dockerfile_dir.is_dir():
            error = create_docker_error(
                f"Dockerfile directory not found: {dockerfile_dir}",
                None,
                None,
                {"dockerfile_dir": str(dockerfile_dir), "image": image_full_name},
            )
            logger.error(
                "dockerfile_directory_invalid", dockerfile_dir=str(dockerfile_dir)
            )
            raise error

        # Check for Dockerfile
        dockerfile_path = dockerfile_dir / "Dockerfile"
        if not dockerfile_path.exists():
            error = create_docker_error(
                f"Dockerfile not found: {dockerfile_path}",
                None,
                None,
                {"dockerfile_path": str(dockerfile_path), "image": image_full_name},
            )
            logger.error("dockerfile_not_found", dockerfile_path=str(dockerfile_path))
            raise error

        # Build the Docker command
        docker_cmd = [
            "docker",
            "build",
            "-t",
            image_full_name,
        ]

        if no_cache:
            docker_cmd.append("--no-cache")

        docker_cmd.append(str(dockerfile_dir))

        # Format command for logging
        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)
        logger.info("docker_build_starting", image=image_full_name)
        logger.debug("docker_command", command=cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            result = await self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("docker_build_executable_not_found", error=str(e))
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Unexpected error building Docker image: {e}",
                cmd_str,
                e,
                {"image": image_full_name, "dockerfile_dir": str(dockerfile_dir)},
            )

            logger.error(
                "docker_build_unexpected_error", image=image_full_name, error=str(e)
            )
            raise error from e

    async def image_exists(self, image_name: str, image_tag: str = "latest") -> bool:
        """Check if a Docker image exists locally."""
        image_full_name = f"{image_name}:{image_tag}"

        # Check Docker availability
        if not await self.is_available():
            logger.warning(
                "docker_not_available_for_image_check", image=image_full_name
            )
            return False

        # Build the Docker command to check image existence
        docker_cmd = ["docker", "inspect", image_full_name]

        try:
            # Run Docker inspect command
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode == 0:
                logger.debug("docker_image_exists", image=image_full_name)
                return True

            # Check if this is a permission error, try with sudo
            stderr_text = stderr.decode() if stderr else ""
            if any(
                phrase in stderr_text.lower()
                for phrase in [
                    "permission denied",
                    "dial unix",
                    "connect: permission denied",
                ]
            ):
                try:
                    logger.debug("docker_image_check_permission_denied_using_sudo")
                    sudo_cmd = ["sudo"] + docker_cmd
                    sudo_process = await asyncio.create_subprocess_exec(
                        *sudo_cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await sudo_process.communicate()
                    if sudo_process.returncode == 0:
                        logger.debug(
                            "docker_image_exists_with_sudo", image=image_full_name
                        )
                        return True
                    else:
                        # Image doesn't exist even with sudo
                        logger.debug(
                            "docker_image_does_not_exist", image=image_full_name
                        )
                        return False
                except Exception:
                    # Image doesn't exist even with sudo
                    logger.debug("Docker image does not exist: %s", image_full_name)
                    return False
            else:
                # Image doesn't exist (inspect returns non-zero exit code)
                logger.debug("Docker image does not exist: %s", image_full_name)
                return False

        except FileNotFoundError:
            logger.warning("docker_image_check_executable_not_found")
            return False

        except Exception as e:
            logger.warning("docker_image_check_unexpected_error", error=str(e))
            return False

    async def pull_image(
        self,
        image_name: str,
        image_tag: str = "latest",
        middleware: OutputMiddleware[T] | None = None,
    ) -> ProcessResult[T]:
        """Pull a Docker image from registry."""

        image_full_name = f"{image_name}:{image_tag}"

        # Check Docker availability
        if not await self.is_available():
            error = create_docker_error(
                "Docker is not available or not properly installed",
                None,
                None,
                {"image": image_full_name},
            )
            logger.error("docker_not_available_for_pull", image=image_full_name)
            raise error

        # Build the Docker command
        docker_cmd = ["docker", "pull", image_full_name]

        # Format command for logging
        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)
        logger.info("docker_pull_starting", image=image_full_name)
        logger.debug("docker_command", command=cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            result = await self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("docker_pull_executable_not_found", error=str(e))
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Unexpected error pulling Docker image: {e}",
                cmd_str,
                e,
                {"image": image_full_name},
            )

            logger.error(
                "docker_pull_unexpected_error", image=image_full_name, error=str(e)
            )
            raise error from e

    # Legacy methods for backward compatibility with plugin system

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

    async def handle_request(
        self, request: Request
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


def create_docker_adapter(
    image: str | None = None,
    volumes: list[DockerVolume] | None = None,
    environment: DockerEnv | None = None,
    additional_args: list[str] | None = None,
    user_context: DockerUserContext | None = None,
) -> DockerAdapterProtocol:
    """
    Factory function to create a DockerAdapter instance.

    Args:
        image: Docker image to use (optional)
        volumes: Optional list of volume mappings
        environment: Optional environment variables
        additional_args: Optional additional Docker arguments
        user_context: Optional user context for container

    Returns:
        Configured DockerAdapter instance

    Example:
        >>> adapter = create_docker_adapter()
        >>> if await adapter.is_available():
        ...     await adapter.run_container("ubuntu:latest", [], {})
    """
    return DockerAdapter()
