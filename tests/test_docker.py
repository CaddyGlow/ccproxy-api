"""Comprehensive tests for Docker functionality.

Tests Docker adapter, path utilities, validation, streaming processes,
and middleware components following the testing patterns from TESTING.md.
"""

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any, Union
from unittest.mock import AsyncMock, patch

import pytest

from ccproxy.docker.adapter import DockerAdapter
from ccproxy.docker.docker_path import DockerPath, DockerPathSet
from ccproxy.docker.middleware import LoggerOutputMiddleware, create_logger_middleware
from ccproxy.docker.models import DockerUserContext
from ccproxy.docker.stream_process import (
    DefaultOutputMiddleware,
    OutputMiddleware,
    run_command,
)
from ccproxy.docker.validators import create_docker_error, validate_port_spec


class TestDockerAdapter:
    """Test DockerAdapter class functionality."""

    async def test_is_available_success(
        self, docker_adapter_success: DockerAdapter
    ) -> None:
        """Test Docker availability check when Docker is available."""
        result = await docker_adapter_success.is_available()
        assert result is True

    async def test_is_available_unavailable(
        self, docker_adapter_unavailable: DockerAdapter
    ) -> None:
        """Test Docker availability check when Docker is not available."""
        result = await docker_adapter_unavailable.is_available()
        assert result is False

    async def test_run_container_success(
        self, docker_adapter_success: DockerAdapter
    ) -> None:
        """Test successful container run operation."""
        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.run_container(
            image="test-image:latest",
            command=["echo", "hello"],
            volumes=[("/host", "/container")],
            environment={"TEST": "value"},
        )

        assert result[0] == 0  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    async def test_run_container_failure(
        self, docker_adapter_failure: DockerAdapter
    ) -> None:
        """Test container run operation failure."""
        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_failure.run_container(
            image="test-image:latest",
            command=["echo", "hello"],
            volumes=[],
            environment={},
        )

        assert result[0] == 1  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    def test_exec_container_success(
        self, docker_adapter_success: DockerAdapter
    ) -> None:
        """Test successful container exec operation."""
        from unittest.mock import patch

        # Mock os.execvp to prevent actual process replacement
        with patch("os.execvp") as mock_execvp:
            docker_adapter_success.exec_container(
                image="test-image:latest",
                command=["ls", "-la"],
                volumes=[],
                environment={},
            )

            # Verify that execvp was called with Docker command
            mock_execvp.assert_called_once()
            args = mock_execvp.call_args[0]
            assert args[0] == "docker"  # First argument should be "docker"
            assert "test-image:latest" in args[1]  # Image should be in command

    async def test_build_image_success(
        self, docker_adapter_success: DockerAdapter, tmp_path: Path
    ) -> None:
        """Test successful image build operation."""
        # Create a mock Dockerfile
        dockerfile_dir = tmp_path / "build_context"
        dockerfile_dir.mkdir()
        dockerfile_path = dockerfile_dir / "Dockerfile"
        dockerfile_path.write_text("FROM alpine:latest\nRUN echo 'test'")

        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.build_image(
            dockerfile_dir=dockerfile_dir, image_name="test-image", image_tag="latest"
        )

        assert result[0] == 0  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    async def test_image_exists_success(
        self, docker_adapter_success: DockerAdapter
    ) -> None:
        """Test checking if image exists."""
        exists = await docker_adapter_success.image_exists("test-image", "latest")
        assert isinstance(exists, bool)

    async def test_pull_image_success(
        self, docker_adapter_success: DockerAdapter
    ) -> None:
        """Test successful image pull operation."""
        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.pull_image("test-image", "latest")

        assert result[0] == 0  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    async def test_run_with_user_context(
        self,
        docker_adapter_success: DockerAdapter,
        docker_user_context: DockerUserContext,
    ) -> None:
        """Test running container with user context."""
        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.run_container(
            image="test-image:latest",
            command=["echo", "hello"],
            volumes=[],
            environment={},
            user_context=docker_user_context,
        )

        assert result[0] == 0  # returncode


class TestDockerPath:
    """Test DockerPath class functionality."""

    def test_docker_path_creation(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath instance creation."""
        assert str(docker_path_fixture.host_path).endswith("host_dir")
        assert docker_path_fixture.container_path == "/app/data"
        assert docker_path_fixture.env_definition_variable_name == "DATA_PATH"

    def test_docker_path_vol(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath volume specification generation."""
        vol_spec = docker_path_fixture.vol()
        assert isinstance(vol_spec, tuple)
        assert len(vol_spec) == 2
        assert vol_spec[1] == "/app/data"

    def test_docker_path_host(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath host path access."""
        host_path = docker_path_fixture.host()
        assert isinstance(host_path, Path)
        assert str(host_path).endswith("host_dir")

    def test_docker_path_container(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath container path access."""
        container_path: Path | str = docker_path_fixture.container()
        assert str(container_path) == "/app/data"

    def test_docker_path_join(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath path joining functionality."""
        joined_path = docker_path_fixture.join("subdir", "file.txt")
        assert joined_path.container_path == "/app/data/subdir/file.txt"
        assert str(joined_path.host_path).endswith("host_dir/subdir/file.txt")

    def test_docker_path_env_definition(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath environment variable definition."""
        env_def = docker_path_fixture.get_env_definition()
        assert "DATA_PATH" in env_def
        assert "/app/data" in env_def

    def test_docker_path_str_repr(self, docker_path_fixture: DockerPath) -> None:
        """Test DockerPath string representations."""
        str_repr = str(docker_path_fixture)
        assert "host_dir" in str_repr
        assert "/app/data" in str_repr

        repr_str = repr(docker_path_fixture)
        assert "DockerPath" in repr_str


class TestDockerPathSet:
    """Test DockerPathSet class functionality."""

    def test_docker_path_set_creation(
        self, docker_path_set_fixture: DockerPathSet
    ) -> None:
        """Test DockerPathSet instance creation."""
        assert len(docker_path_set_fixture.paths) == 2

        data1_path = docker_path_set_fixture.get("data1")
        data2_path = docker_path_set_fixture.get("data2")
        assert data1_path.container_path == "/app/data1"
        assert data2_path.container_path == "/app/data2"

    def test_docker_path_set_has_method(
        self, docker_path_set_fixture: DockerPathSet
    ) -> None:
        """Test DockerPathSet has method."""
        assert docker_path_set_fixture.has("data1") is True
        assert docker_path_set_fixture.has("data2") is True
        assert docker_path_set_fixture.has("nonexistent") is False

    def test_docker_path_set_names(
        self, docker_path_set_fixture: DockerPathSet
    ) -> None:
        """Test DockerPathSet names method."""
        names = docker_path_set_fixture.names()
        assert "data1" in names
        assert "data2" in names
        assert len(names) == 2

    def test_docker_path_set_volumes(
        self, docker_path_set_fixture: DockerPathSet
    ) -> None:
        """Test DockerPathSet volume generation."""
        volumes = docker_path_set_fixture.volumes()
        assert len(volumes) == 2

        for vol_spec in volumes:
            assert isinstance(vol_spec, tuple)
            assert len(vol_spec) == 2


class TestDockerModels:
    """Test Docker model classes."""

    def test_docker_user_context_creation(
        self, docker_user_context: DockerUserContext
    ) -> None:
        """Test DockerUserContext model creation."""
        assert docker_user_context.uid == 1000
        assert docker_user_context.gid == 1000
        assert docker_user_context.username == "testuser"
        assert docker_user_context.enable_user_mapping is True

    def test_docker_user_context_validation(self) -> None:
        """Test DockerUserContext validation."""
        # Test with minimal required fields
        context = DockerUserContext.create_manual(uid=1001, gid=1001, username="test2")
        assert context.uid == 1001
        assert context.gid == 1001
        assert context.username == "test2"

    def test_docker_user_context_docker_flag(
        self, docker_user_context: DockerUserContext
    ) -> None:
        """Test DockerUserContext Docker user flag generation."""
        flag = docker_user_context.get_docker_user_flag()
        assert flag == "1000:1000"

    def test_docker_user_context_volumes(
        self, docker_user_context: DockerUserContext
    ) -> None:
        """Test DockerUserContext volume generation."""
        volumes = docker_user_context.get_volumes()
        assert isinstance(volumes, list)

    def test_docker_user_context_environment(
        self, docker_user_context: DockerUserContext
    ) -> None:
        """Test DockerUserContext environment variable generation."""
        env_vars = docker_user_context.get_environment_variables()
        assert isinstance(env_vars, dict)


class TestDockerValidators:
    """Test Docker validation functions."""

    def test_validate_port_spec_valid_single_port(self) -> None:
        """Test port spec validation with valid port mapping."""
        result = validate_port_spec("8080:80")
        assert result == "8080:80"

    def test_validate_port_spec_valid_port_mapping(self) -> None:
        """Test port spec validation with valid port mapping."""
        result = validate_port_spec("8080:80")
        assert result == "8080:80"

    def test_validate_port_spec_valid_host_ip(self) -> None:
        """Test port spec validation with host IP."""
        result = validate_port_spec("127.0.0.1:8080:80")
        assert result == "127.0.0.1:8080:80"

    def test_validate_port_spec_invalid_format(self) -> None:
        """Test port spec validation with invalid format."""
        from ccproxy.core.errors import DockerError

        with pytest.raises(DockerError, match="Invalid port specification"):
            validate_port_spec("invalid:port:spec:too:many")

    def test_validate_port_spec_invalid_port_number(self) -> None:
        """Test port spec validation with invalid port number."""
        from ccproxy.core.errors import DockerError

        with pytest.raises(DockerError, match="Invalid port numbers"):
            validate_port_spec("99999:80")

    def test_validate_port_spec_invalid_port_zero(self) -> None:
        """Test port spec validation with zero port."""
        from ccproxy.core.errors import DockerError

        with pytest.raises(DockerError, match="Invalid port numbers"):
            validate_port_spec("0:80")

    def test_create_docker_error(self) -> None:
        """Test Docker error creation."""
        error = create_docker_error("Test error message", "docker run test")
        assert "Test error message" in str(error)


class TestStreamProcess:
    """Test streaming process functionality."""

    async def test_default_output_middleware(
        self, output_middleware: DefaultOutputMiddleware
    ) -> None:
        """Test DefaultOutputMiddleware functionality."""
        assert isinstance(output_middleware, DefaultOutputMiddleware)

        # Test processing output
        result = await output_middleware.process("test output", "stdout")
        assert result == "test output"

    async def test_logger_output_middleware(self) -> None:
        """Test LoggerOutputMiddleware functionality."""
        from structlog import get_logger

        logger = get_logger("test")
        middleware = LoggerOutputMiddleware(logger)

        # Test processing output - should return the same data
        result = await middleware.process("test output", "stdout")
        assert result == "test output"

    def test_create_logger_middleware(self) -> None:
        """Test logger middleware creation."""
        middleware = create_logger_middleware()
        assert isinstance(middleware, LoggerOutputMiddleware)

    async def test_run_command_success(self) -> None:
        """Test successful command execution."""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        # Mock the streams
        mock_stdout = AsyncMock()
        mock_stdout.readline.side_effect = [b"command output\n", b""]
        mock_stderr = AsyncMock()
        mock_stderr.readline.side_effect = [b""]

        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result: tuple[int, list[str], list[str]] = await run_command(
                ["echo", "test"]
            )

        assert result[0] == 0  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    async def test_run_command_failure(self) -> None:
        """Test command execution failure."""

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1

        # Mock the streams
        mock_stdout = AsyncMock()
        mock_stdout.readline.side_effect = [b""]
        mock_stderr = AsyncMock()
        mock_stderr.readline.side_effect = [b"command failed\n", b""]

        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result: tuple[int, list[str], list[str]] = await run_command(["false"])

        assert result[0] == 1  # returncode
        assert isinstance(result[1], list)  # stdout
        assert isinstance(result[2], list)  # stderr

    async def test_run_command_with_middleware(self) -> None:
        """Test command execution with output middleware."""

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        # Mock the streams
        mock_stdout = AsyncMock()
        mock_stdout.readline.side_effect = [b"output\n", b""]
        mock_stderr = AsyncMock()
        mock_stderr.readline.side_effect = [b""]

        mock_proc.stdout = mock_stdout
        mock_proc.stderr = mock_stderr

        middleware = DefaultOutputMiddleware()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await run_command(["echo", "test"], middleware=middleware)

        assert result[0] == 0  # returncode
        assert isinstance(result[1], list)  # stdout

    def test_chained_middleware_creation(self) -> None:
        """Test creation of chained middleware."""
        from structlog import get_logger

        from ccproxy.docker.middleware import create_chained_docker_middleware

        logger = get_logger("test")
        middleware1 = LoggerOutputMiddleware(logger)
        middleware2 = DefaultOutputMiddleware()

        chained = create_chained_docker_middleware([middleware1, middleware2])
        assert chained is not None

    async def test_middleware_process_chain(self) -> None:
        """Test middleware processing chain."""
        from ccproxy.docker.stream_process import ChainedOutputMiddleware

        middleware1 = DefaultOutputMiddleware()
        middleware2 = DefaultOutputMiddleware()

        chained: ChainedOutputMiddleware[str] = ChainedOutputMiddleware(
            [middleware1, middleware2]
        )
        result = await chained.process("test data", "stdout")

        # Should process through both middleware instances
        assert result == "test data"


class TestDockerIntegration:
    """Integration tests for Docker components."""

    async def test_adapter_with_path_integration(
        self,
        docker_adapter_success: DockerAdapter,
        docker_path_fixture: DockerPath,
    ) -> None:
        """Test DockerAdapter integration with DockerPath."""
        vol_spec = docker_path_fixture.vol()
        volumes = [vol_spec]

        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.run_container(
            image="test-image:latest",
            command=["ls", docker_path_fixture.container_path],
            volumes=volumes,
            environment={},
        )

        assert result[0] == 0  # returncode

    async def test_adapter_with_user_context_integration(
        self,
        docker_adapter_success: DockerAdapter,
        docker_user_context: DockerUserContext,
    ) -> None:
        """Test DockerAdapter integration with DockerUserContext."""
        result: tuple[
            int, list[str], list[str]
        ] = await docker_adapter_success.run_container(
            image="test-image:latest",
            command=["echo", "integration test"],
            volumes=[],
            environment={},
            user_context=docker_user_context,
        )

        assert result[0] == 0  # returncode

    def test_path_set_with_context_integration(
        self,
        docker_path_set_fixture: DockerPathSet,
        docker_user_context: DockerUserContext,
    ) -> None:
        """Test DockerPathSet integration with DockerUserContext."""
        volumes = docker_path_set_fixture.volumes()
        names = docker_path_set_fixture.names()

        # Verify we can integrate path set data into user context
        assert len(volumes) == 2
        assert len(names) == 2

        # Verify path names
        assert "data1" in names
        assert "data2" in names
