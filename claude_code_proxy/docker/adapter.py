"""Docker adapter for container operations."""

import datetime
import hashlib
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from claude_code_proxy.exceptions import DockerError
from claude_code_proxy.utils.logging import get_logger

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
    create_chained_middleware,
    run_command,
)


logger = get_logger(__name__)


def validate_port_spec(port_spec: str) -> str:
    """Validate a Docker port specification string.

    Supports formats like:
    - "8080:80"
    - "localhost:8080:80"
    - "127.0.0.1:8080:80"
    - "8080:80/tcp"
    - "localhost:8080:80/udp"
    - "[::1]:8080:80"

    Args:
        port_spec: Port specification string

    Returns:
        Validated port specification string

    Raises:
        DockerError: If port specification is invalid
    """
    if not port_spec or not isinstance(port_spec, str):
        raise create_docker_error(
            f"Invalid port specification: {port_spec!r}",
            details={"port_spec": port_spec},
        )

    # Remove protocol suffix for validation if present
    port_part = port_spec
    protocol = None
    if "/" in port_spec:
        port_part, protocol = port_spec.rsplit("/", 1)
        if protocol not in ("tcp", "udp"):
            raise create_docker_error(
                f"Invalid protocol in port specification: {protocol}",
                details={"port_spec": port_spec, "protocol": protocol},
            )

    # Handle IPv6 address format specially
    if port_part.startswith("["):
        # IPv6 format like [::1]:8080:80
        ipv6_end = port_part.find("]:")
        if ipv6_end == -1:
            raise create_docker_error(
                f"Invalid IPv6 port specification format: {port_spec}",
                details={
                    "port_spec": port_spec,
                    "expected_format": "[ipv6]:host_port:container_port",
                },
            )

        host_ip = port_part[: ipv6_end + 1]  # Include the closing ]
        remaining = port_part[ipv6_end + 2 :]  # Skip ]:
        port_parts = remaining.split(":")

        if len(port_parts) != 2:
            raise create_docker_error(
                f"Invalid IPv6 port specification format: {port_spec}",
                details={
                    "port_spec": port_spec,
                    "expected_format": "[ipv6]:host_port:container_port",
                },
            )

        host_port, container_port = port_parts
        parts = [host_ip, host_port, container_port]
    else:
        # Regular format
        parts = port_part.split(":")

    if len(parts) == 2:
        # Format: "host_port:container_port"
        host_port, container_port = parts
        try:
            host_port_num = int(host_port)
            container_port_num = int(container_port)
            if not (1 <= host_port_num <= 65535) or not (
                1 <= container_port_num <= 65535
            ):
                raise ValueError("Port numbers must be between 1 and 65535")
        except ValueError as e:
            raise create_docker_error(
                f"Invalid port numbers in specification: {port_spec}",
                details={"port_spec": port_spec, "error": str(e)},
            ) from e

    elif len(parts) == 3:
        # Format: "host_ip:host_port:container_port"
        host_ip, host_port, container_port = parts

        # Basic IP validation (simplified)
        if not host_ip or host_ip in (
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "[::1]",
        ):
            pass  # Common valid values
        elif host_ip.startswith("[") and host_ip.endswith("]"):
            pass  # IPv6 format like [::1]
        else:
            # Basic check for IPv4-like format
            ip_parts = host_ip.split(".")
            if len(ip_parts) == 4:
                try:
                    for part in ip_parts:
                        num = int(part)
                        if not (0 <= num <= 255):
                            raise ValueError("Invalid IPv4 address")
                except ValueError as e:
                    raise create_docker_error(
                        f"Invalid host IP in port specification: {host_ip}",
                        details={
                            "port_spec": port_spec,
                            "host_ip": host_ip,
                            "error": str(e),
                        },
                    ) from e

        try:
            host_port_num = int(host_port)
            container_port_num = int(container_port)
            if not (1 <= host_port_num <= 65535) or not (
                1 <= container_port_num <= 65535
            ):
                raise ValueError("Port numbers must be between 1 and 65535")
        except ValueError as e:
            raise create_docker_error(
                f"Invalid port numbers in specification: {port_spec}",
                details={"port_spec": port_spec, "error": str(e)},
            ) from e
    else:
        raise create_docker_error(
            f"Invalid port specification format: {port_spec}",
            details={
                "port_spec": port_spec,
                "expected_format": "host_port:container_port or host_ip:host_port:container_port",
            },
        )

    return port_spec


class LoggerOutputMiddleware(OutputMiddleware[str]):
    """Simple middleware that prints output with optional prefixes.

    This middleware prints each line to the console with configurable
    prefixes for stdout and stderr streams.
    """

    def __init__(
        self, logger: logging.Logger, stdout_prefix: str = "", stderr_prefix: str = ""
    ):
        """Initialize middleware with custom prefixes.

        Args:
            stdout_prefix: Prefix for stdout lines (default: "")
            stderr_prefix: Prefix for stderr lines (default: "")
        """
        self.logger = logger
        self.stderr_prefix = stderr_prefix
        self.stdout_prefix = stdout_prefix

    def process(self, line: str, stream_type: str) -> str:
        """Process and print a line with the appropriate prefix.

        Args:
            line: Output line to process
            stream_type: Either "stdout" or "stderr"

        Returns:
            The original line (unmodified)
        """
        if stream_type == "stdout":
            logger.info(f"{self.stdout_prefix}{line}")
            # logger.debug(f"{self.stdout_prefix}{line}")
        else:
            logger.info(f"{self.stderr_prefix}{line}")
            # logger.info(f"{self.stderr_prefix}{line}")
        return line


class DockerAdapter:
    """Implementation of Docker adapter."""

    def _needs_sudo(self) -> bool:
        """Check if Docker requires sudo by testing docker info command."""
        try:
            subprocess.run(
                ["docker", "info"], check=True, capture_output=True, text=True
            )
            return False
        except subprocess.CalledProcessError as e:
            # Check if error suggests permission issues
            return e.stderr and (
                "permission denied" in e.stderr.lower()
                or "dial unix" in e.stderr.lower()
                or "connect: permission denied" in e.stderr.lower()
            )
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if Docker is available on the system."""
        docker_cmd = ["docker", "--version"]
        cmd_str = " ".join(docker_cmd)

        try:
            result = subprocess.run(
                docker_cmd, check=True, capture_output=True, text=True
            )
            docker_version = result.stdout.strip()
            logger.debug("Docker is available: %s", docker_version)
            return True

        except FileNotFoundError:
            logger.warning("Docker executable not found in PATH")
            return False

        except subprocess.CalledProcessError as e:
            stderr = e.stderr if hasattr(e, "stderr") and e.stderr else "unknown error"
            logger.warning("Docker command failed: %s - error: %s", cmd_str, stderr)
            return False

        except Exception as e:
            logger.warning("Unexpected error checking Docker availability: %s", e)
            return False

    def _run_with_sudo_fallback(
        self, docker_cmd: list[str], middleware: OutputMiddleware[T]
    ) -> ProcessResult[T]:
        # Try without sudo first
        try:
            result = run_command(docker_cmd, middleware)
            return result
        except subprocess.SubprocessError as e:
            # Check if this might be a permission error
            if hasattr(e, "stderr") and e.stderr:
                stderr_text = str(e.stderr).lower()
                if any(
                    phrase in stderr_text
                    for phrase in [
                        "permission denied",
                        "dial unix",
                        "connect: permission denied",
                    ]
                ):
                    logger.info("Docker permission denied, trying with sudo...")
                    sudo_cmd = ["sudo"] + docker_cmd
                    return run_command(sudo_cmd, middleware)
            # Re-raise if not a permission error
            raise

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
    ) -> ProcessResult[T]:
        """Run a Docker container with specified configuration."""

        docker_cmd = ["docker", "run", "--rm"]

        # Add user context if provided and should be used
        if user_context and user_context.should_use_user_mapping():
            docker_user_flag = user_context.get_docker_user_flag()
            docker_cmd.extend(["--user", docker_user_flag])
            logger.debug("Using Docker user mapping: %s", docker_user_flag)

        # Add custom entrypoint if specified
        if entrypoint:
            docker_cmd.extend(["--entrypoint", entrypoint])
            logger.debug("Using custom entrypoint: %s", entrypoint)

        # Add port publishing if specified
        if ports:
            for port_spec in ports:
                validated_port = validate_port_spec(port_spec)
                docker_cmd.extend(["-p", validated_port])
                logger.debug("Publishing port: %s", validated_port)

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
        logger.debug("Docker command: %s", cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            # Try with sudo fallback if needed
            result = self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("Docker executable not found: %s", e)
            raise error from e

        except subprocess.SubprocessError as e:
            error = create_docker_error(f"Docker subprocess error: {e}", cmd_str, e)
            logger.error("Docker subprocess error: %s", e)
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
            logger.error("Unexpected error running Docker container: %s", e)
            raise error from e

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
            logger.debug("Using Docker user mapping: %s", docker_user_flag)

        # Add custom entrypoint if specified
        if entrypoint:
            docker_cmd.extend(["--entrypoint", entrypoint])
            logger.debug("Using custom entrypoint: %s", entrypoint)

        # Add port publishing if specified
        if ports:
            for port_spec in ports:
                validated_port = validate_port_spec(port_spec)
                docker_cmd.extend(["-p", validated_port])
                logger.debug("Publishing port: %s", validated_port)

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
        logger.info("Executing Docker command with execvp: %s", cmd_str)

        try:
            # Check if we need sudo (without running the actual command)
            if self._needs_sudo():
                logger.info("Docker requires sudo, using sudo for execution")
                docker_cmd = ["sudo"] + docker_cmd

            # Replace current process with Docker command
            os.execvp(docker_cmd[0], docker_cmd)

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("Docker executable not found for execvp: %s", e)
            raise error from e

        except OSError as e:
            error = create_docker_error(
                f"Failed to execute Docker command: {e}", cmd_str, e
            )
            logger.error("OSError during Docker execvp: %s", e)
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
            logger.error("Unexpected error during Docker execvp: %s", e)
            raise error from e

    def build_image(
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
        if not self.is_available():
            error = create_docker_error(
                "Docker is not available or not properly installed",
                None,
                None,
                {"image": image_full_name},
            )
            logger.error("Docker not available for image build: %s", image_full_name)
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
                "Invalid Dockerfile directory for image build: %s", dockerfile_dir
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
            logger.error("Dockerfile not found at %s for image build", dockerfile_path)
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
        logger.info("Building Docker image: %s", image_full_name)
        logger.debug("Docker command: %s", cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            result = self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("Docker executable not found during image build: %s", e)
            raise error from e

        except subprocess.SubprocessError as e:
            error = create_docker_error(f"Docker subprocess error: {e}", cmd_str, e)
            logger.error("Docker subprocess error: %s", e)
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Unexpected error building Docker image: {e}",
                cmd_str,
                e,
                {"image": image_full_name, "dockerfile_dir": str(dockerfile_dir)},
            )

            logger.error("Unexpected Docker build error for %s: %s", image_full_name, e)
            raise error from e

    def image_exists(self, image_name: str, image_tag: str = "latest") -> bool:
        """Check if a Docker image exists locally."""
        image_full_name = f"{image_name}:{image_tag}"

        # Check Docker availability
        if not self.is_available():
            logger.warning(
                "Docker not available, cannot check image existence: %s",
                image_full_name,
            )
            return False

        # Build the Docker command to check image existence
        docker_cmd = ["docker", "inspect", image_full_name]
        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)

        try:
            # Run Docker inspect command
            result = subprocess.run(
                docker_cmd, check=True, capture_output=True, text=True
            )

            logger.debug("Docker image exists: %s", image_full_name)
            return True

        except subprocess.CalledProcessError as e:
            # Check if this is a permission error, try with sudo
            if e.stderr and any(
                phrase in e.stderr.lower()
                for phrase in [
                    "permission denied",
                    "dial unix",
                    "connect: permission denied",
                ]
            ):
                try:
                    logger.debug(
                        "Docker permission denied, trying with sudo for image check..."
                    )
                    sudo_cmd = ["sudo"] + docker_cmd
                    subprocess.run(sudo_cmd, check=True, capture_output=True, text=True)
                    logger.debug("Docker image exists (with sudo): %s", image_full_name)
                    return True
                except subprocess.CalledProcessError:
                    # Image doesn't exist even with sudo
                    logger.debug("Docker image does not exist: %s", image_full_name)
                    return False
            else:
                # Image doesn't exist (inspect returns non-zero exit code)
                logger.debug("Docker image does not exist: %s", image_full_name)
                return False

        except FileNotFoundError:
            logger.warning("Docker executable not found during image check")
            return False

        except Exception as e:
            logger.warning("Unexpected error checking Docker image existence: %s", e)
            return False

    def pull_image(
        self,
        image_name: str,
        image_tag: str = "latest",
        middleware: OutputMiddleware[T] | None = None,
    ) -> ProcessResult[T]:
        """Pull a Docker image from registry."""

        image_full_name = f"{image_name}:{image_tag}"

        # Check Docker availability
        if not self.is_available():
            error = create_docker_error(
                "Docker is not available or not properly installed",
                None,
                None,
                {"image": image_full_name},
            )
            logger.error("Docker not available for image pull: %s", image_full_name)
            raise error

        # Build the Docker command
        docker_cmd = ["docker", "pull", image_full_name]

        # Format command for logging
        cmd_str = " ".join(shlex.quote(arg) for arg in docker_cmd)
        logger.info("Pulling Docker image: %s", image_full_name)
        logger.debug("Docker command: %s", cmd_str)

        try:
            if middleware is None:
                # Cast is needed because T is unbound at this point
                middleware = cast(OutputMiddleware[T], LoggerOutputMiddleware(logger))

            result = self._run_with_sudo_fallback(docker_cmd, middleware)

            return result

        except FileNotFoundError as e:
            error = create_docker_error(f"Docker executable not found: {e}", cmd_str, e)
            logger.error("Docker executable not found during image pull: %s", e)
            raise error from e

        except subprocess.SubprocessError as e:
            error = create_docker_error(f"Docker subprocess error: {e}", cmd_str, e)
            logger.error("Docker subprocess error: %s", e)
            raise error from e

        except Exception as e:
            error = create_docker_error(
                f"Unexpected error pulling Docker image: {e}",
                cmd_str,
                e,
                {"image": image_full_name},
            )

            logger.error("Unexpected Docker pull error for %s: %s", image_full_name, e)
            raise error from e


def create_logger_middleware(
    logger_instance: logging.Logger | None = None,
    stdout_prefix: str = "",
    stderr_prefix: str = "",
) -> LoggerOutputMiddleware:
    """Factory function to create a LoggerOutputMiddleware instance.

    Args:
        logger_instance: Logger instance to use (defaults to module logger)
        stdout_prefix: Prefix for stdout lines
        stderr_prefix: Prefix for stderr lines

    Returns:
        Configured LoggerOutputMiddleware instance
    """
    if logger_instance is None:
        logger_instance = logger
    return LoggerOutputMiddleware(logger_instance, stdout_prefix, stderr_prefix)


def create_chained_docker_middleware(
    middleware_chain: list[OutputMiddleware[Any]],
    include_logger: bool = True,
    logger_instance: logging.Logger | None = None,
    stdout_prefix: str = "",
    stderr_prefix: str = "",
) -> OutputMiddleware[Any]:
    """Factory function to create chained middleware for Docker operations.

    Args:
        middleware_chain: List of middleware components to chain together
        include_logger: Whether to automatically add logger middleware at the end
        logger_instance: Logger instance to use (defaults to module logger)
        stdout_prefix: Prefix for stdout lines in logger middleware
        stderr_prefix: Prefix for stderr lines in logger middleware

    Returns:
        Chained middleware instance

    """
    final_chain = list(middleware_chain)

    if include_logger:
        logger_middleware = create_logger_middleware(
            logger_instance, stdout_prefix, stderr_prefix
        )
        final_chain.append(logger_middleware)

    if len(final_chain) == 1:
        return final_chain[0]

    return create_chained_middleware(final_chain)


def create_docker_error(
    message: str,
    command: str | None = None,
    cause: Exception | None = None,
    details: dict[str, Any] | None = None,
) -> DockerError:
    """Create a DockerError with standardized context.

    Args:
        message: Human-readable error message
        command: Docker command that failed (optional)
        cause: Original exception that caused this error (optional)
        details: Additional context details (optional)

    Returns:
        DockerError instance with all context information
    """
    return DockerError(
        message=message,
        command=command,
        cause=cause,
        details=details,
    )


def create_docker_adapter() -> DockerAdapterProtocol:
    """
    Factory function to create a DockerAdapter instance.

    Returns:
        Configured DockerAdapter instance

    Example:
        >>> adapter = create_docker_adapter()
        >>> if adapter.is_available():
        ...     adapter.run_container("ubuntu:latest", [], {})
    """
    return DockerAdapter()
