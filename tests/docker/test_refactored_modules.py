"""Tests for refactored Docker modules to ensure proper imports and functionality."""

import logging
from unittest.mock import Mock, patch

import pytest

from claude_code_proxy.docker import (
    DockerAdapter,
    LoggerOutputMiddleware,
    create_chained_docker_middleware,
    create_docker_adapter,
    create_docker_error,
    create_logger_middleware,
    validate_port_spec,
)
from claude_code_proxy.docker.middleware import (
    LoggerOutputMiddleware as MiddlewareLoggerOutputMiddleware,
)
from claude_code_proxy.docker.stream_process import OutputMiddleware
from claude_code_proxy.docker.validators import (
    create_docker_error as validators_create_docker_error,
)
from claude_code_proxy.docker.validators import (
    validate_port_spec as validators_validate_port_spec,
)
from claude_code_proxy.exceptions import DockerError


class TestModuleImports:
    """Test that all imports work correctly after refactoring."""

    def test_adapter_imports(self):
        """Test imports from adapter module."""
        assert DockerAdapter is not None
        assert create_docker_adapter is not None

    def test_middleware_imports(self):
        """Test imports from middleware module."""
        assert LoggerOutputMiddleware is not None
        assert create_logger_middleware is not None
        assert create_chained_docker_middleware is not None
        # Verify it's the same class
        assert LoggerOutputMiddleware is MiddlewareLoggerOutputMiddleware

    def test_validators_imports(self):
        """Test imports from validators module."""
        assert validate_port_spec is not None
        assert create_docker_error is not None
        # Verify they're the same functions
        assert validate_port_spec is validators_validate_port_spec
        assert create_docker_error is validators_create_docker_error


class TestValidators:
    """Test validator functions after refactoring."""

    def test_validate_port_spec_simple(self):
        """Test simple port specification validation."""
        assert validate_port_spec("8080:80") == "8080:80"
        assert validate_port_spec("3000:3000") == "3000:3000"

    def test_validate_port_spec_with_host(self):
        """Test port specification with host."""
        assert validate_port_spec("localhost:8080:80") == "localhost:8080:80"
        assert validate_port_spec("127.0.0.1:8080:80") == "127.0.0.1:8080:80"
        assert validate_port_spec("0.0.0.0:8080:80") == "0.0.0.0:8080:80"

    def test_validate_port_spec_with_protocol(self):
        """Test port specification with protocol."""
        assert validate_port_spec("8080:80/tcp") == "8080:80/tcp"
        assert validate_port_spec("8080:80/udp") == "8080:80/udp"

    def test_validate_port_spec_ipv6(self):
        """Test IPv6 port specification."""
        assert validate_port_spec("[::1]:8080:80") == "[::1]:8080:80"

    def test_validate_port_spec_invalid(self):
        """Test invalid port specifications."""
        with pytest.raises(DockerError) as exc_info:
            validate_port_spec("")
        assert "Invalid port specification" in str(exc_info.value)

        with pytest.raises(DockerError) as exc_info:
            validate_port_spec("8080")
        assert "Invalid port specification format" in str(exc_info.value)

        with pytest.raises(DockerError) as exc_info:
            validate_port_spec("65536:80")
        assert "Invalid port numbers" in str(exc_info.value)

    def test_create_docker_error(self):
        """Test Docker error creation."""
        error = create_docker_error("Test error")
        assert isinstance(error, DockerError)
        assert error.message == "Test error"
        assert error.error_type == "docker_error"
        assert error.status_code == 500

        # With all parameters
        cause = ValueError("Original error")
        error = create_docker_error(
            "Test error with context",
            command="docker run test",
            cause=cause,
            details={"key": "value"},
        )
        assert error.message == "Test error with context"
        assert error.details["command"] == "docker run test"
        assert error.details["cause"] == "Original error"
        assert error.details["cause_type"] == "ValueError"
        assert error.details["key"] == "value"


class TestMiddleware:
    """Test middleware functions after refactoring."""

    def test_logger_output_middleware(self):
        """Test LoggerOutputMiddleware functionality."""
        mock_logger = Mock(spec=logging.Logger)
        middleware = LoggerOutputMiddleware(
            mock_logger, stdout_prefix="OUT: ", stderr_prefix="ERR: "
        )

        # Test stdout processing
        result = middleware.process("test stdout", "stdout")
        assert result == "test stdout"
        mock_logger.info.assert_called_with("OUT: test stdout")

        # Test stderr processing
        result = middleware.process("test stderr", "stderr")
        assert result == "test stderr"
        mock_logger.info.assert_called_with("ERR: test stderr")

    def test_create_logger_middleware(self):
        """Test logger middleware factory."""
        mock_logger = Mock(spec=logging.Logger)
        middleware = create_logger_middleware(
            logger_instance=mock_logger,
            stdout_prefix="[STDOUT] ",
            stderr_prefix="[STDERR] ",
        )

        assert isinstance(middleware, LoggerOutputMiddleware)
        assert middleware.logger is mock_logger
        assert middleware.stdout_prefix == "[STDOUT] "
        assert middleware.stderr_prefix == "[STDERR] "

    def test_create_logger_middleware_default(self):
        """Test logger middleware factory with defaults."""
        middleware = create_logger_middleware()
        assert isinstance(middleware, LoggerOutputMiddleware)
        assert middleware.stdout_prefix == ""
        assert middleware.stderr_prefix == ""

    def test_create_chained_docker_middleware(self):
        """Test chained middleware creation."""
        mock_middleware1 = Mock(spec=OutputMiddleware)
        mock_middleware2 = Mock(spec=OutputMiddleware)

        # Without logger
        chained = create_chained_docker_middleware(
            [mock_middleware1, mock_middleware2], include_logger=False
        )
        assert chained is not None

        # With logger
        chained = create_chained_docker_middleware(
            [mock_middleware1], include_logger=True
        )
        assert chained is not None

        # Single middleware without logger returns it directly
        chained = create_chained_docker_middleware(
            [mock_middleware1], include_logger=False
        )
        assert chained is mock_middleware1


class TestDockerAdapter:
    """Test DockerAdapter still works after refactoring."""

    def test_docker_adapter_creation(self):
        """Test DockerAdapter can be instantiated."""
        adapter = DockerAdapter()
        assert adapter is not None

    def test_create_docker_adapter(self):
        """Test docker adapter factory."""
        adapter = create_docker_adapter()
        assert isinstance(adapter, DockerAdapter)

    @patch("subprocess.run")
    def test_is_available(self, mock_run):
        """Test Docker availability check."""
        # Test when Docker is available
        mock_run.return_value = Mock(
            returncode=0, stdout="Docker version 20.10.0", stderr=""
        )
        adapter = DockerAdapter()
        assert adapter.is_available() is True

        # Test when Docker is not available
        mock_run.side_effect = FileNotFoundError("docker not found")
        assert adapter.is_available() is False
