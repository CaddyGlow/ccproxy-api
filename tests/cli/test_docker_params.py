"""Unit tests for Docker parameter definitions."""

import tempfile
from unittest.mock import Mock

import pytest
import typer

from ccproxy.cli.docker.params import (
    DockerOptions,
    docker_arg_option,
    docker_env_option,
    docker_home_option,
    docker_image_option,
    docker_volume_option,
    docker_workspace_option,
    user_gid_option,
    user_mapping_option,
    user_uid_option,
)
from ccproxy.utils.docker_params import (
    parse_docker_env,
    parse_docker_volume,
    validate_docker_arg,
    validate_docker_home,
    validate_docker_image,
    validate_docker_workspace,
    validate_user_gid,
    validate_user_uid,
)


# Create mock objects for Typer callbacks
def create_mock_context() -> Mock:
    """Create a mock Typer context."""
    return Mock(spec=typer.Context)


def create_mock_param() -> Mock:
    """Create a mock Typer callback parameter."""
    return Mock(spec=typer.CallbackParam)


def test_docker_options_class():
    """Test DockerOptions class structure."""
    # Check that DockerOptions can be instantiated
    options = DockerOptions()

    # Check default values
    assert options.docker_image is None
    assert options.docker_env == []
    assert options.docker_volume == []
    assert options.docker_arg == []
    assert options.docker_home is None
    assert options.docker_workspace is None
    assert options.user_mapping_enabled is None
    assert options.user_uid is None
    assert options.user_gid is None

    # Check with custom values
    custom_options = DockerOptions(
        docker_image="test:latest",
        docker_env=["KEY=value"],
        docker_volume=["/host:/container"],
        docker_arg=["--privileged"],
        docker_home="/home",
        docker_workspace="/workspace",
        user_mapping_enabled=True,
        user_uid=1000,
        user_gid=1000,
    )

    assert custom_options.docker_image == "test:latest"
    assert custom_options.docker_env == ["KEY=value"]
    assert custom_options.docker_volume == ["/host:/container"]
    assert custom_options.docker_arg == ["--privileged"]
    assert custom_options.docker_home == "/home"
    assert custom_options.docker_workspace == "/workspace"
    assert custom_options.user_mapping_enabled is True
    assert custom_options.user_uid == 1000
    assert custom_options.user_gid == 1000


def test_docker_param_functions_exist():
    """Test that all Docker parameter functions exist and are callable."""
    # Test that all functions exist and are callable
    assert callable(docker_image_option)
    assert callable(docker_env_option)
    assert callable(docker_volume_option)
    assert callable(docker_arg_option)
    assert callable(docker_home_option)
    assert callable(docker_workspace_option)
    assert callable(user_mapping_option)
    assert callable(user_uid_option)
    assert callable(user_gid_option)


def test_validate_docker_image():
    """Test docker image validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid images
    assert validate_docker_image(ctx, param, "ubuntu:latest") == "ubuntu:latest"
    assert validate_docker_image(ctx, param, "python:3.9-slim") == "python:3.9-slim"
    assert validate_docker_image(ctx, param, None) is None

    # Invalid images
    with pytest.raises(Exception) as exc_info:
        validate_docker_image(ctx, param, "ubuntu invalid")
    assert "spaces" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        validate_docker_image(ctx, param, "")
    assert "empty" in str(exc_info.value)


def test_parse_docker_env():
    """Test docker environment variable parsing."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid env vars
    assert parse_docker_env(ctx, param, ["KEY=value"]) == ["KEY=value"]
    assert parse_docker_env(ctx, param, ["A=1", "B=2"]) == ["A=1", "B=2"]
    assert parse_docker_env(ctx, param, []) == []

    # Invalid env vars
    with pytest.raises(Exception) as exc_info:
        parse_docker_env(ctx, param, ["INVALID"])
    assert "KEY=VALUE" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        parse_docker_env(ctx, param, ["=value"])
    assert "Key cannot be empty" in str(exc_info.value)


def test_parse_docker_volume():
    """Test docker volume parsing."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Valid volumes
        result = parse_docker_volume(ctx, param, [f"{tmpdir}:/container"])
        assert len(result) == 1
        assert f"{tmpdir}:/container" in result[0]

        # Test with options
        result = parse_docker_volume(ctx, param, [f"{tmpdir}:/container:ro"])
        assert len(result) == 1
        assert f"{tmpdir}:/container:ro" in result[0]

        assert parse_docker_volume(ctx, param, []) == []

    # Invalid volumes
    with pytest.raises(Exception) as exc_info:
        parse_docker_volume(ctx, param, ["invalid"])
    assert "host:container" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        parse_docker_volume(ctx, param, [":/container"])
    assert "Host path cannot be empty" in str(exc_info.value)

    with pytest.raises(Exception) as exc_info:
        parse_docker_volume(ctx, param, ["/nonexistent:/container"])
    assert "does not exist" in str(exc_info.value)


def test_validate_docker_arg():
    """Test docker argument validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid args
    assert validate_docker_arg(ctx, param, ["--privileged"]) == ["--privileged"]
    assert validate_docker_arg(ctx, param, ["--network=host", "-v"]) == [
        "--network=host",
        "-v",
    ]
    assert validate_docker_arg(ctx, param, []) == []

    # Invalid args
    with pytest.raises(Exception) as exc_info:
        validate_docker_arg(ctx, param, [""])
    assert "empty" in str(exc_info.value)


def test_validate_docker_home():
    """Test docker home directory validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid paths
    assert validate_docker_home(ctx, param, "/home/user") == "/home/user"
    assert validate_docker_home(ctx, param, None) is None

    # Relative paths should be converted to absolute
    result = validate_docker_home(ctx, param, "relative/path")
    assert result is not None and result.startswith("/")  # Should be absolute


def test_validate_docker_workspace():
    """Test docker workspace directory validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid paths
    assert validate_docker_workspace(ctx, param, "/workspace") == "/workspace"
    assert validate_docker_workspace(ctx, param, None) is None

    # Relative paths should be converted to absolute
    result = validate_docker_workspace(ctx, param, "workspace")
    assert result is not None and result.startswith("/")  # Should be absolute


def test_validate_user_uid():
    """Test user UID validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid UIDs
    assert validate_user_uid(ctx, param, 0) == 0
    assert validate_user_uid(ctx, param, 1000) == 1000
    assert validate_user_uid(ctx, param, None) is None

    # Invalid UIDs
    with pytest.raises(Exception) as exc_info:
        validate_user_uid(ctx, param, -1)
    assert "non-negative" in str(exc_info.value)


def test_validate_user_gid():
    """Test user GID validation."""
    ctx = create_mock_context()
    param = create_mock_param()

    # Valid GIDs
    assert validate_user_gid(ctx, param, 0) == 0
    assert validate_user_gid(ctx, param, 1000) == 1000
    assert validate_user_gid(ctx, param, None) is None

    # Invalid GIDs
    with pytest.raises(Exception) as exc_info:
        validate_user_gid(ctx, param, -1)
    assert "non-negative" in str(exc_info.value)
