"""Tests for the new Docker integration module."""

from pathlib import Path

import pytest

from claude_code_proxy.docker import (
    DockerAdapter,
    DockerPath,
    DockerUserContext,
    create_docker_adapter,
    create_docker_error,
    validate_port_spec,
)
from claude_code_proxy.exceptions import DockerError


class TestDockerPath:
    """Test DockerPath functionality."""

    def test_docker_path_creation(self):
        """Test creating a DockerPath instance."""
        path = DockerPath(
            host_path=Path("/host/path"), container_path="/container/path"
        )

        assert path.host_path == Path("/host/path")
        assert path.container_path == "/container/path"

    def test_docker_path_vol(self):
        """Test getting volume mapping from DockerPath."""
        path = DockerPath(
            host_path=Path("/host/path"), container_path="/container/path"
        )

        host_str, container_str = path.vol()
        assert host_str == str(Path("/host/path"))
        assert container_str == "/container/path"

    def test_docker_path_container(self):
        """Test getting container path."""
        path = DockerPath(
            host_path=Path("/host/path"), container_path="/container/path"
        )

        assert path.container() == "/container/path"

    def test_docker_path_join(self):
        """Test joining paths in DockerPath."""
        path = DockerPath(host_path=Path("/host"), container_path="/container")

        joined = path.join("subdir", "file.txt")
        assert joined.host_path == Path("/host/subdir/file.txt")
        assert joined.container_path == "/container/subdir/file.txt"


class TestDockerUserContext:
    """Test DockerUserContext functionality."""

    def test_docker_user_context_creation(self):
        """Test creating a DockerUserContext."""
        home_path = DockerPath(
            host_path=Path("/home/user"), container_path="/data/home"
        )

        ctx = DockerUserContext(
            uid=1000, gid=1000, username="testuser", home_path=home_path
        )

        assert ctx.uid == 1000
        assert ctx.gid == 1000
        assert ctx.username == "testuser"
        assert ctx.home_path == home_path

    def test_get_docker_user_flag(self):
        """Test getting Docker user flag."""
        ctx = DockerUserContext(uid=1000, gid=1000, username="testuser")

        assert ctx.get_docker_user_flag() == "1000:1000"

    def test_get_volumes(self):
        """Test getting volume mappings."""
        home_path = DockerPath(
            host_path=Path("/home/user"), container_path="/data/home"
        )
        workspace_path = DockerPath(
            host_path=Path("/workspace"), container_path="/data/workspace"
        )

        ctx = DockerUserContext(
            uid=1000,
            gid=1000,
            username="testuser",
            home_path=home_path,
            workspace_path=workspace_path,
        )

        volumes = ctx.get_volumes()
        assert len(volumes) == 2
        assert (str(Path("/home/user")), "/data/home") in volumes
        assert (str(Path("/workspace")), "/data/workspace") in volumes

    def test_get_environment_variables(self):
        """Test getting environment variables."""
        home_path = DockerPath(
            host_path=Path("/home/user"), container_path="/data/home"
        )
        workspace_path = DockerPath(
            host_path=Path("/workspace"), container_path="/data/workspace"
        )

        ctx = DockerUserContext(
            uid=1000,
            gid=1000,
            username="testuser",
            home_path=home_path,
            workspace_path=workspace_path,
        )

        env = ctx.get_environment_variables()
        assert env["HOME"] == "/data/home"
        assert env["CLAUDE_HOME"] == "/data/home"
        assert env["CLAUDE_WORKSPACE"] == "/data/workspace"


class TestPortValidation:
    """Test port specification validation."""

    def test_validate_simple_port_spec(self):
        """Test validating simple port specifications."""
        assert validate_port_spec("8080:80") == "8080:80"
        assert validate_port_spec("3000:3000") == "3000:3000"

    def test_validate_host_ip_port_spec(self):
        """Test validating port specs with host IP."""
        assert validate_port_spec("localhost:8080:80") == "localhost:8080:80"
        assert validate_port_spec("127.0.0.1:8080:80") == "127.0.0.1:8080:80"
        assert validate_port_spec("192.168.1.100:8080:80") == "192.168.1.100:8080:80"

    def test_validate_port_spec_with_protocol(self):
        """Test validating port specs with protocol."""
        assert validate_port_spec("8080:80/tcp") == "8080:80/tcp"
        assert validate_port_spec("localhost:8080:80/udp") == "localhost:8080:80/udp"

    def test_validate_ipv6_port_spec(self):
        """Test validating IPv6 port specifications."""
        assert validate_port_spec("[::1]:8080:80") == "[::1]:8080:80"

    def test_invalid_port_specs(self):
        """Test that invalid port specs raise errors."""
        with pytest.raises(DockerError):
            validate_port_spec("")

        with pytest.raises(DockerError):
            validate_port_spec("invalid")

        with pytest.raises(DockerError):
            validate_port_spec("8080:80:90:100")

        with pytest.raises(DockerError):
            validate_port_spec("8080:80/invalid")

        with pytest.raises(DockerError):
            validate_port_spec("99999:80")  # Port out of range


class TestDockerAdapter:
    """Test DockerAdapter functionality."""

    def test_create_docker_adapter(self):
        """Test creating a Docker adapter."""
        adapter = create_docker_adapter()
        assert isinstance(adapter, DockerAdapter)

    def test_docker_adapter_availability_check(self):
        """Test Docker availability check (doesn't require Docker to be installed)."""
        adapter = create_docker_adapter()
        # We don't assert the result since Docker may or may not be available
        # in the test environment, but the method should not crash
        availability = adapter.is_available()
        assert isinstance(availability, bool)

    def test_exec_container_method_exists(self):
        """Test that exec_container method exists with correct signature."""
        adapter = create_docker_adapter()

        # Check that the method exists
        assert hasattr(adapter, "exec_container")
        assert callable(adapter.exec_container)

        # We can't actually test exec_container because it would replace
        # the current process, but we can verify it has the right signature
        import inspect

        sig = inspect.signature(adapter.exec_container)
        expected_params = {
            "image",
            "volumes",
            "environment",
            "command",
            "user_context",
            "entrypoint",
            "ports",
        }
        actual_params = set(sig.parameters.keys())
        assert expected_params.issubset(actual_params)


class TestDockerError:
    """Test Docker error creation."""

    def test_create_docker_error(self):
        """Test creating Docker errors."""
        error = create_docker_error(
            "Test error message",
            command="docker run test",
            details={"image": "test-image"},
        )

        assert isinstance(error, DockerError)
        assert error.message == "Test error message"
        assert error.details["command"] == "docker run test"
        assert error.details["image"] == "test-image"

    def test_create_docker_error_with_cause(self):
        """Test creating Docker errors with cause."""
        original_error = ValueError("Original error")
        docker_error = create_docker_error(
            "Docker operation failed", cause=original_error
        )

        assert docker_error.details["cause"] == "Original error"
        assert docker_error.details["cause_type"] == "ValueError"
