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
    parse_docker_env,
    parse_docker_volume,
    user_gid_option,
    user_mapping_option,
    user_uid_option,
    validate_docker_arg,
    validate_docker_home,
    validate_docker_image,
    validate_docker_workspace,
    validate_user_gid,
    validate_user_uid,
)


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

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid images
    assert validate_docker_image(ctx, param, "ubuntu:latest") == "ubuntu:latest"
    assert validate_docker_image(ctx, param, "python:3.9-slim") == "python:3.9-slim"
    assert validate_docker_image(ctx, param, None) is None

    # Invalid images
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_docker_image(ctx, param, "ubuntu invalid")
    assert "spaces" in str(exc_info.value)

    with pytest.raises(typer.BadParameter) as exc_info:
        validate_docker_image(ctx, param, "")
    assert "empty" in str(exc_info.value)


def test_parse_docker_env():
    """Test docker environment variable parsing."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid env vars
    result = parse_docker_env(ctx, param, ["KEY=value"])
    assert result == ["KEY=value"]

    result = parse_docker_env(ctx, param, ["API_KEY=secret123", "DEBUG=true"])
    assert result == ["API_KEY=secret123", "DEBUG=true"]

    result = parse_docker_env(ctx, param, [])
    assert result == []

    # Invalid env vars
    with pytest.raises(typer.BadParameter) as exc_info:
        parse_docker_env(ctx, param, ["INVALID"])
    assert "Expected KEY=VALUE" in str(exc_info.value)

    with pytest.raises(typer.BadParameter) as exc_info:
        parse_docker_env(ctx, param, [""])
    assert "Expected KEY=VALUE" in str(exc_info.value)

    # "=value" is actually valid - it creates a key with empty name
    result = parse_docker_env(ctx, param, ["=value"])
    assert result == ["=value"]


def test_parse_docker_volume():
    """Test docker volume parsing."""
    from unittest.mock import Mock

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

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

        # Empty list
        result = parse_docker_volume(ctx, param, [])
        assert result == []

    # Invalid volumes
    with pytest.raises(typer.BadParameter) as exc_info:
        parse_docker_volume(ctx, param, ["invalid"])
    assert "host:container" in str(exc_info.value)

    with pytest.raises(typer.BadParameter) as exc_info:
        parse_docker_volume(ctx, param, [":/container"])
    assert "host:container" in str(exc_info.value)


def test_validate_docker_arg():
    """Test docker argument validation."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid args
    result = validate_docker_arg(ctx, param, ["--privileged"])
    assert result == ["--privileged"]

    result = validate_docker_arg(ctx, param, ["--network=host", "-v"])
    assert result == ["--network=host", "-v"]

    result = validate_docker_arg(ctx, param, [])
    assert result == []


def test_validate_docker_home():
    """Test docker home directory validation."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid paths
    result = validate_docker_home(ctx, param, "/home/user")
    assert result == "/home/user"
    assert validate_docker_home(ctx, param, None) is None

    # Relative paths should be converted to absolute
    result = validate_docker_home(ctx, param, "relative/path")
    assert result is not None and result.startswith("/")  # Should be absolute


def test_validate_docker_workspace():
    """Test docker workspace directory validation."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid paths
    result = validate_docker_workspace(ctx, param, "/workspace")
    assert result == "/workspace"
    assert validate_docker_workspace(ctx, param, None) is None

    # Relative paths should be converted to absolute
    result = validate_docker_workspace(ctx, param, "workspace")
    assert result is not None and result.startswith("/")  # Should be absolute


def test_validate_user_uid():
    """Test user UID validation."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid UIDs
    assert validate_user_uid(ctx, param, 0) == 0
    assert validate_user_uid(ctx, param, 1000) == 1000
    assert validate_user_uid(ctx, param, None) is None

    # Invalid UIDs
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_user_uid(ctx, param, -1)
    assert "non-negative" in str(exc_info.value)


def test_validate_user_gid():
    """Test user GID validation."""

    ctx = Mock(spec=typer.Context)
    param = Mock(spec=typer.CallbackParam)

    # Valid GIDs
    assert validate_user_gid(ctx, param, 0) == 0
    assert validate_user_gid(ctx, param, 1000) == 1000
    assert validate_user_gid(ctx, param, None) is None

    # Invalid GIDs
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_user_gid(ctx, param, -1)
    assert "non-negative" in str(exc_info.value)
