"""Tests for Docker settings module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ccproxy.config.docker_settings import DockerSettings


@pytest.mark.unit
class TestDockerSettings:
    """Test DockerSettings class."""

    def test_docker_settings_defaults(self) -> None:
        """Test DockerSettings default values."""
        settings = DockerSettings()

        assert settings.docker_image.startswith(
            "ghcr.io/caddyglow/claude-code-proxy-api:"
        )
        # Default volumes are created by setup_docker_volumes
        assert len(settings.docker_volumes) == 2
        assert any("/data/home" in v for v in settings.docker_volumes)
        assert any("/data/workspace" in v for v in settings.docker_volumes)
        assert "CLAUDE_HOME" in settings.docker_environment
        assert "CLAUDE_WORKSPACE" in settings.docker_environment
        assert settings.docker_additional_args == []
        assert settings.docker_home_directory is None
        assert settings.docker_workspace_directory is None
        assert settings.user_mapping_enabled is True

    def test_docker_settings_custom_values(self, tmp_path: Path) -> None:
        """Test DockerSettings with custom values."""
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "workspace"
        home_dir.mkdir()
        workspace_dir.mkdir()

        settings = DockerSettings(
            docker_image="custom-image:latest",
            docker_volumes=[f"{tmp_path}:/data"],
            docker_environment={"CUSTOM_VAR": "value"},
            docker_additional_args=["--privileged"],
            docker_home_directory=str(home_dir),
            docker_workspace_directory=str(workspace_dir),
            user_mapping_enabled=False,
            user_uid=1001,
            user_gid=1001,
        )

        assert settings.docker_image == "custom-image:latest"
        assert f"{tmp_path}:/data" in settings.docker_volumes
        assert settings.docker_environment["CUSTOM_VAR"] == "value"
        assert "--privileged" in settings.docker_additional_args
        assert settings.docker_home_directory == str(home_dir)
        assert settings.docker_workspace_directory == str(workspace_dir)
        assert settings.user_mapping_enabled is False
        assert settings.user_uid == 1001
        assert settings.user_gid == 1001

    def test_docker_volume_validation_success(self, tmp_path: Path) -> None:
        """Test successful Docker volume validation."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        settings = DockerSettings(
            docker_volumes=[f"{test_dir}:/app/data", f"{test_dir}:/app/config:ro"]
        )

        assert len(settings.docker_volumes) == 2
        assert f"{test_dir}:/app/data" in settings.docker_volumes
        assert f"{test_dir}:/app/config:ro" in settings.docker_volumes

    def test_docker_volume_validation_failure(self) -> None:
        """Test Docker volume validation failure."""
        with pytest.raises(ValueError, match="Invalid volume format"):
            DockerSettings(docker_volumes=["invalid_volume_format"])

        with pytest.raises(ValueError, match="Host path does not exist"):
            DockerSettings(docker_volumes=["/nonexistent/path:/app/data"])

    def test_docker_home_directory_validation(self, tmp_path: Path) -> None:
        """Test Docker home directory validation."""
        home_dir = tmp_path / "home"
        home_dir.mkdir()

        settings = DockerSettings(docker_home_directory=str(home_dir))
        assert settings.docker_home_directory == str(home_dir)

        # Test with relative path
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            settings = DockerSettings(docker_home_directory="home")
            assert settings.docker_home_directory == str(home_dir)
        finally:
            os.chdir(original_cwd)

    def test_docker_workspace_directory_validation(self, tmp_path: Path) -> None:
        """Test Docker workspace directory validation."""
        workspace_dir = tmp_path / "workspace"
        workspace_dir.mkdir()

        settings = DockerSettings(docker_workspace_directory=str(workspace_dir))
        assert settings.docker_workspace_directory == str(workspace_dir)

    def test_setup_docker_volumes_default(self) -> None:
        """Test default Docker volume setup."""
        settings = DockerSettings()

        # Default volumes should be created
        assert len(settings.docker_volumes) == 2
        assert any("/data/home" in v for v in settings.docker_volumes)
        assert any("/data/workspace" in v for v in settings.docker_volumes)

    def test_setup_docker_volumes_with_custom_directories(self, tmp_path: Path) -> None:
        """Test Docker volume setup with custom directories."""
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "workspace"
        home_dir.mkdir()
        workspace_dir.mkdir()

        settings = DockerSettings(
            docker_home_directory=str(home_dir),
            docker_workspace_directory=str(workspace_dir),
        )

        # No default volumes should be created when custom directories are set
        assert len(settings.docker_volumes) == 0

    def test_setup_docker_volumes_with_explicit_volumes(self, tmp_path: Path) -> None:
        """Test Docker volume setup with explicit volumes."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        settings = DockerSettings(docker_volumes=[f"{test_dir}:/custom"])

        # Explicit volumes should be preserved
        assert len(settings.docker_volumes) == 1
        assert f"{test_dir}:/custom" in settings.docker_volumes

    def test_environment_variables_setup(self) -> None:
        """Test environment variables are set up correctly."""
        settings = DockerSettings()

        assert settings.docker_environment["CLAUDE_HOME"] == "/data/home"
        assert settings.docker_environment["CLAUDE_WORKSPACE"] == "/data/workspace"

    def test_environment_variables_not_overridden(self) -> None:
        """Test that existing environment variables are not overridden."""
        settings = DockerSettings(
            docker_environment={
                "CLAUDE_HOME": "/custom/home",
                "CLAUDE_WORKSPACE": "/custom/workspace",
            }
        )

        assert settings.docker_environment["CLAUDE_HOME"] == "/custom/home"
        assert settings.docker_environment["CLAUDE_WORKSPACE"] == "/custom/workspace"

    @patch("os.name", "posix")
    @patch("os.getuid", return_value=1234)
    @patch("os.getgid", return_value=5678)
    def test_user_mapping_auto_detection_unix(self, mock_getgid, mock_getuid) -> None:
        """Test auto-detection of UID/GID on Unix systems."""
        settings = DockerSettings(user_mapping_enabled=True)

        assert settings.user_mapping_enabled is True
        assert settings.user_uid == 1234
        assert settings.user_gid == 5678

    def test_user_mapping_auto_detection_windows(self) -> None:
        """Test user mapping behavior on Windows systems."""
        # Skip this test on non-Windows systems due to WindowsPath issues
        if os.name != "nt":
            pytest.skip("Windows-specific test")

        settings = DockerSettings(user_mapping_enabled=True)
        # On Windows, user mapping should be automatically disabled
        assert settings.user_mapping_enabled is False

    def test_user_mapping_explicit_values(self) -> None:
        """Test user mapping with explicit values."""
        settings = DockerSettings(
            user_mapping_enabled=True,
            user_uid=2001,
            user_gid=2002,
        )

        assert settings.user_mapping_enabled is True
        assert settings.user_uid == 2001
        assert settings.user_gid == 2002

    def test_user_mapping_validation(self) -> None:
        """Test user mapping UID/GID validation."""
        # Valid values
        settings = DockerSettings(user_uid=0, user_gid=0)
        assert settings.user_uid == 0
        assert settings.user_gid == 0

        # Invalid values (negative)
        with pytest.raises(ValueError):
            DockerSettings(user_uid=-1)

        with pytest.raises(ValueError):
            DockerSettings(user_gid=-1)

    def test_complex_configuration(self, tmp_path: Path) -> None:
        """Test complex Docker settings configuration."""
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        home_dir.mkdir()
        workspace_dir.mkdir()
        data_dir.mkdir()

        settings = DockerSettings(
            docker_image="complex-image:v2.0",
            docker_volumes=[
                f"{data_dir}:/app/data:ro",
                f"{home_dir}:/app/home",
            ],
            docker_environment={
                "APP_ENV": "production",
                "DEBUG": "false",
                "CUSTOM_HOME": "/app/home",
            },
            docker_additional_args=["--cap-add=SYS_PTRACE", "--memory=4g"],
            docker_home_directory=str(home_dir),
            docker_workspace_directory=str(workspace_dir),
            user_mapping_enabled=True,
            user_uid=3001,
            user_gid=3002,
        )

        assert settings.docker_image == "complex-image:v2.0"
        assert len(settings.docker_volumes) == 2
        assert f"{data_dir}:/app/data:ro" in settings.docker_volumes
        assert f"{home_dir}:/app/home" in settings.docker_volumes
        assert settings.docker_environment["APP_ENV"] == "production"
        assert settings.docker_environment["DEBUG"] == "false"
        assert settings.docker_environment["CUSTOM_HOME"] == "/app/home"
        assert "--cap-add=SYS_PTRACE" in settings.docker_additional_args
        assert "--memory=4g" in settings.docker_additional_args
        assert settings.user_uid == 3001
        assert settings.user_gid == 3002
