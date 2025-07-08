"""Tests for Docker validation utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from ccproxy.utils.docker_validation import (
    validate_docker_volumes,
    validate_environment_variable,
    validate_host_path,
    validate_volume_format,
)


class TestValidateVolumeFormat:
    """Tests for validate_volume_format function."""

    def test_valid_volume_format_absolute_path(self, tmp_path: Path) -> None:
        """Test validation with absolute path."""
        # Create a temporary directory
        host_path = tmp_path / "test_dir"
        host_path.mkdir()
        volume = f"{host_path}:/container/path"

        result = validate_volume_format(volume)
        assert result == f"{host_path}:/container/path"

    def test_valid_volume_format_with_options(self, tmp_path: Path) -> None:
        """Test validation with volume options."""
        host_path = tmp_path / "test_dir"
        host_path.mkdir()
        volume = f"{host_path}:/container/path:ro"

        result = validate_volume_format(volume)
        assert result == f"{host_path}:/container/path:ro"

    def test_valid_volume_format_relative_path(self, tmp_path: Path) -> None:
        """Test validation with relative path that gets converted to absolute."""
        # Create a directory in tmp_path
        host_path = tmp_path / "test_dir"
        host_path.mkdir()

        # Change to tmp_path so relative path works
        original_cwd = str(Path.cwd())
        try:
            os.chdir(tmp_path)
            volume = "test_dir:/container/path"

            result = validate_volume_format(volume)
            assert result == f"{host_path}:/container/path"
        finally:
            os.chdir(original_cwd)

    def test_valid_volume_format_with_env_var(self, tmp_path: Path) -> None:
        """Test validation with environment variable in path."""
        host_path = tmp_path / "test_dir"
        host_path.mkdir()

        # Set environment variable
        os.environ["TEST_DIR"] = str(host_path)
        try:
            volume = "$TEST_DIR:/container/path"

            result = validate_volume_format(volume)
            assert result == f"{host_path}:/container/path"
        finally:
            del os.environ["TEST_DIR"]

    def test_invalid_volume_format_no_colon(self) -> None:
        """Test validation fails when colon is missing."""
        with pytest.raises(
            ValueError, match="Invalid volume format.*Expected 'host:container"
        ):
            validate_volume_format("/path/without/colon")

    def test_invalid_volume_format_empty_parts(self) -> None:
        """Test validation fails with empty parts."""
        with pytest.raises(
            ValueError, match="Invalid volume format.*Expected 'host:container"
        ):
            validate_volume_format(":")

    def test_invalid_volume_format_nonexistent_path(self) -> None:
        """Test validation fails when host path doesn't exist."""
        with pytest.raises(ValueError, match="Host path does not exist"):
            validate_volume_format("/nonexistent/path:/container/path")

    def test_volume_with_multiple_colons(self, tmp_path: Path) -> None:
        """Test volume with multiple colons (including Windows-style paths)."""
        host_path = tmp_path / "test_dir"
        host_path.mkdir()

        # Test with options
        volume = f"{host_path}:/container/path:rw,z"
        result = validate_volume_format(volume)
        assert result == f"{host_path}:/container/path:rw,z"


class TestValidateHostPath:
    """Tests for validate_host_path function."""

    def test_valid_absolute_path(self) -> None:
        """Test validation with absolute path."""
        path = "/usr/local/bin"
        result = validate_host_path(path)
        assert result == path

    def test_valid_relative_path(self) -> None:
        """Test validation with relative path that gets converted to absolute."""
        path = "."
        result = validate_host_path(path)
        assert Path(result).is_absolute()
        assert result == str(Path.cwd())

    def test_path_with_env_var(self, tmp_path: Path) -> None:
        """Test validation with environment variable."""
        os.environ["TEST_PATH"] = str(tmp_path)
        try:
            path = "$TEST_PATH/subdir"
            result = validate_host_path(path)
            assert result == f"{tmp_path}/subdir"
        finally:
            del os.environ["TEST_PATH"]

    def test_home_directory_expansion(self) -> None:
        """Test validation with home directory symbol."""
        path = "~/test"
        result = validate_host_path(path)
        assert Path(result).is_absolute()
        assert not result.startswith("~")

    def test_complex_relative_path(self, tmp_path: Path) -> None:
        """Test validation with complex relative path."""
        # Create nested directories
        nested_dir = tmp_path / "a" / "b" / "c"
        nested_dir.mkdir(parents=True)

        original_cwd = str(Path.cwd())
        try:
            os.chdir(tmp_path)
            path = "./a/b/../b/c"
            result = validate_host_path(path)
            assert Path(result).is_absolute()
            assert result == str(nested_dir)
        finally:
            os.chdir(original_cwd)


class TestValidateEnvironmentVariable:
    """Tests for validate_environment_variable function."""

    def test_valid_env_var(self) -> None:
        """Test validation with valid environment variable."""
        env_var = "KEY=value"
        key, value = validate_environment_variable(env_var)
        assert key == "KEY"
        assert value == "value"

    def test_env_var_with_equals_in_value(self) -> None:
        """Test validation with equals sign in value."""
        env_var = "KEY=value=with=equals"
        key, value = validate_environment_variable(env_var)
        assert key == "KEY"
        assert value == "value=with=equals"

    def test_env_var_with_empty_value(self) -> None:
        """Test validation with empty value."""
        env_var = "KEY="
        key, value = validate_environment_variable(env_var)
        assert key == "KEY"
        assert value == ""

    def test_env_var_with_spaces(self) -> None:
        """Test validation with spaces in value."""
        env_var = "KEY=value with spaces"
        key, value = validate_environment_variable(env_var)
        assert key == "KEY"
        assert value == "value with spaces"

    def test_invalid_env_var_no_equals(self) -> None:
        """Test validation fails when equals sign is missing."""
        with pytest.raises(
            ValueError, match="Invalid environment variable format.*Expected KEY=VALUE"
        ):
            validate_environment_variable("KEYVALUE")

    def test_env_var_with_special_chars(self) -> None:
        """Test validation with special characters."""
        env_var = "MY_KEY_123=value!@#$%^&*()"
        key, value = validate_environment_variable(env_var)
        assert key == "MY_KEY_123"
        assert value == "value!@#$%^&*()"


class TestValidateDockerVolumes:
    """Tests for validate_docker_volumes function."""

    def test_valid_volumes_list(self, tmp_path: Path) -> None:
        """Test validation with multiple valid volumes."""
        # Create multiple directories
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        volumes = [
            f"{dir1}:/container/dir1",
            f"{dir2}:/container/dir2:ro",
        ]

        result = validate_docker_volumes(volumes)
        assert len(result) == 2
        assert result[0] == f"{dir1}:/container/dir1"
        assert result[1] == f"{dir2}:/container/dir2:ro"

    def test_empty_volumes_list(self) -> None:
        """Test validation with empty list."""
        result = validate_docker_volumes([])
        assert result == []

    def test_volumes_list_with_relative_paths(self, tmp_path: Path) -> None:
        """Test validation with relative paths that get converted."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        original_cwd = str(Path.cwd())
        try:
            os.chdir(tmp_path)
            volumes = ["dir1:/container/dir1"]

            result = validate_docker_volumes(volumes)
            assert len(result) == 1
            assert result[0] == f"{dir1}:/container/dir1"
        finally:
            os.chdir(original_cwd)

    def test_volumes_list_with_invalid_volume(self, tmp_path: Path) -> None:
        """Test validation fails when one volume is invalid."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        volumes = [
            f"{dir1}:/container/dir1",
            "invalid_volume",  # Missing colon
        ]

        with pytest.raises(ValueError, match="Invalid volume format"):
            validate_docker_volumes(volumes)

    def test_volumes_list_with_nonexistent_path(self, tmp_path: Path) -> None:
        """Test validation fails when path doesn't exist."""
        dir1 = tmp_path / "dir1"
        dir1.mkdir()

        volumes = [
            f"{dir1}:/container/dir1",
            "/nonexistent/path:/container/path",
        ]

        with pytest.raises(ValueError, match="Host path does not exist"):
            validate_docker_volumes(volumes)
