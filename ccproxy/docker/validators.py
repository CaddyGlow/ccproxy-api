"""Docker validation utilities and error creation."""

from typing import Any

from ccproxy.exceptions import DockerError


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
