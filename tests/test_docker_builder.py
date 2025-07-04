"""Test Docker command builder utility."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_code_proxy.config.settings import DockerSettings
from claude_code_proxy.utils.docker_builder import DockerCommandBuilder


@pytest.mark.unit
class TestDockerCommandBuilder:
    """Test DockerCommandBuilder class."""

    @pytest.fixture
    def basic_docker_settings(self) -> DockerSettings:
        """Create basic Docker settings for testing."""
        return DockerSettings(
            docker_image="test-image:latest",
            docker_volumes=[],
            docker_environment={},
            docker_additional_args=[],
            docker_home_directory=None,
            docker_workspace_directory=None,
        )

    @pytest.fixture
    def configured_docker_settings(self, tmp_path: Path) -> DockerSettings:
        """Create Docker settings with configuration for testing."""
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        home_dir.mkdir()
        workspace_dir.mkdir()
        data_dir.mkdir()

        return DockerSettings(
            docker_image="configured-image:v1.0",
            docker_volumes=[f"{data_dir}:/app/data"],
            docker_environment={"ENV_VAR": "test_value", "DEBUG": "true"},
            docker_additional_args=["--privileged", "--network=host"],
            docker_home_directory=str(home_dir),
            docker_workspace_directory=str(workspace_dir),
        )

    def test_init(self, basic_docker_settings: DockerSettings) -> None:
        """Test DockerCommandBuilder initialization."""
        builder = DockerCommandBuilder(basic_docker_settings)
        assert builder.settings == basic_docker_settings

    def test_build_command_basic(self, basic_docker_settings: DockerSettings) -> None:
        """Test basic Docker command building."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]

        cmd = builder.build_command(claude_args)

        # Check basic command structure
        assert cmd[:4] == ["docker", "run", "--rm", "-it"]
        assert "test-image:latest" in cmd
        assert "claude" in cmd
        assert "--help" in cmd

        # Check environment variables are set
        env_indices = [i for i, arg in enumerate(cmd) if arg == "--env"]
        assert len(env_indices) >= 2  # At least CLAUDE_HOME and CLAUDE_WORKSPACE

    def test_build_command_with_docker_image_override(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test Docker command building with image override."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--version"]

        cmd = builder.build_command(claude_args, docker_image="override-image:latest")

        assert "override-image:latest" in cmd
        assert "test-image:latest" not in cmd

    def test_build_command_with_configured_settings(
        self, configured_docker_settings: DockerSettings
    ) -> None:
        """Test Docker command building with configured settings."""
        builder = DockerCommandBuilder(configured_docker_settings)
        claude_args = ["chat", "Hello"]

        cmd = builder.build_command(claude_args)

        # Check basic command structure
        assert cmd[:4] == ["docker", "run", "--rm", "-it"]
        assert "configured-image:v1.0" in cmd
        assert "claude" in cmd
        assert "chat" in cmd
        assert "Hello" in cmd

        # Check volumes are included
        volume_indices = [i for i, arg in enumerate(cmd) if arg == "--volume"]
        assert len(volume_indices) >= 2  # At least home and workspace

        # Check environment variables
        env_indices = [i for i, arg in enumerate(cmd) if arg == "--env"]
        assert len(env_indices) >= 4  # At least 4 env vars

        # Check additional args
        assert "--privileged" in cmd
        assert "--network=host" in cmd

    def test_build_command_with_docker_env_override(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test Docker command building with environment variable overrides."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]
        docker_env = ["CUSTOM_VAR=custom_value", "ANOTHER_VAR=another_value"]

        cmd = builder.build_command(claude_args, docker_env=docker_env)

        assert "--env" in cmd
        assert "CUSTOM_VAR=custom_value" in cmd
        assert "ANOTHER_VAR=another_value" in cmd

    def test_build_command_with_invalid_env_format(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test Docker command building with invalid environment variable format."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]
        docker_env = ["INVALID_FORMAT"]

        with pytest.raises(ValueError, match="Invalid environment variable format"):
            builder.build_command(claude_args, docker_env=docker_env)

    def test_build_command_with_docker_volume_override(
        self, basic_docker_settings: DockerSettings, tmp_path: Path
    ) -> None:
        """Test Docker command building with volume overrides."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]

        # Create a temporary directory for the volume test
        test_dir = tmp_path / "test_volume"
        test_dir.mkdir()

        docker_volume = [f"{test_dir}:/app/test"]

        cmd = builder.build_command(claude_args, docker_volume=docker_volume)

        assert "--volume" in cmd
        volume_mount = f"{test_dir}:/app/test"
        assert volume_mount in cmd

    def test_build_command_with_docker_arg_override(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test Docker command building with additional Docker arguments."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]
        docker_arg = ["--user", "1000:1000", "--memory", "2g"]

        cmd = builder.build_command(claude_args, docker_arg=docker_arg)

        assert "--user" in cmd
        assert "1000:1000" in cmd
        assert "--memory" in cmd
        assert "2g" in cmd

    def test_build_command_with_custom_directories(
        self, basic_docker_settings: DockerSettings, tmp_path: Path
    ) -> None:
        """Test Docker command building with custom home and workspace directories."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]

        home_dir = tmp_path / "custom_home"
        workspace_dir = tmp_path / "custom_workspace"
        home_dir.mkdir()
        workspace_dir.mkdir()

        cmd = builder.build_command(
            claude_args,
            docker_home=str(home_dir),
            docker_workspace=str(workspace_dir),
        )

        # Check that custom volumes are created
        assert "--volume" in cmd
        expected_home_volume = f"{home_dir}:/data/home"
        expected_workspace_volume = f"{workspace_dir}:/data/workspace"
        assert expected_home_volume in cmd
        assert expected_workspace_volume in cmd

    def test_validate_volume_format_valid(
        self, basic_docker_settings: DockerSettings, tmp_path: Path
    ) -> None:
        """Test volume format validation with valid formats."""
        builder = DockerCommandBuilder(basic_docker_settings)

        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Test various valid formats
        valid_volumes = [
            f"{test_dir}:/app/data",
            f"{test_dir}:/app/data:ro",
            f"{test_dir}:/app/data:rw,Z",
        ]

        for volume in valid_volumes:
            result = builder._validate_volume_format(volume)
            assert result.startswith(str(test_dir))
            assert ":/app/data" in result

    def test_validate_volume_format_invalid(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test volume format validation with invalid formats."""
        builder = DockerCommandBuilder(basic_docker_settings)

        invalid_volumes = [
            "invalid_format",  # No colon
            "",  # Empty string
        ]

        for volume in invalid_volumes:
            with pytest.raises(ValueError, match="Invalid volume format"):
                builder._validate_volume_format(volume)

        # Test case where container path is missing
        with pytest.raises(ValueError):  # Could be either format or path error
            builder._validate_volume_format("/nonexistent/path:")

    def test_validate_volume_format_nonexistent_path(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test volume format validation with nonexistent host path."""
        builder = DockerCommandBuilder(basic_docker_settings)

        nonexistent_path = "/definitely/does/not/exist"
        volume = f"{nonexistent_path}:/app/data"

        with pytest.raises(ValueError, match="Host path does not exist"):
            builder._validate_volume_format(volume)

    def test_validate_volume_format_relative_path(
        self, basic_docker_settings: DockerSettings, tmp_path: Path
    ) -> None:
        """Test volume format validation with relative paths."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Create a test directory
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        # Change working directory and use a relative path
        import os
        from pathlib import Path

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            volume = "test:/app/data"
            result = builder._validate_volume_format(volume)

            # Should convert to absolute path
            assert result.startswith(str(test_dir))
            assert ":/app/data" in result
        finally:
            os.chdir(original_cwd)

    def test_validate_host_path(self, basic_docker_settings: DockerSettings) -> None:
        """Test host path validation and normalization."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Test absolute path
        abs_path = "/home/user"
        result = builder._validate_host_path(abs_path)
        assert result == abs_path

        # Test path with environment variables
        with patch.dict(os.environ, {"HOME": "/home/user"}):
            env_path = "$HOME/documents"
            result = builder._validate_host_path(env_path)
            assert result == "/home/user/documents"

    def test_validate_working_directory(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test working directory validation and normalization."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Test absolute path
        abs_path = "/app/workspace"
        result = builder._validate_working_directory(abs_path)
        assert result == abs_path

        # Test relative path conversion
        import os
        from pathlib import Path

        original_cwd = Path.cwd()
        try:
            os.chdir("/tmp")
            rel_path = "workspace"
            result = builder._validate_working_directory(rel_path)
            assert "/tmp/workspace" in result
        finally:
            os.chdir(original_cwd)

    def test_get_merged_volumes(
        self, basic_docker_settings: DockerSettings, tmp_path: Path
    ) -> None:
        """Test volume merging functionality."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Set up base volumes
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_volumes = [f"{base_dir}:/app/base"]

        # Set up CLI volumes
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        cli_volumes = [f"{cli_dir}:/app/cli"]

        result = builder._get_merged_volumes(cli_volumes, base_volumes)

        assert len(result) == 2
        assert f"{base_dir}:/app/base" in result
        assert f"{cli_dir}:/app/cli" in result

    def test_get_merged_environment(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test environment variable merging functionality."""
        builder = DockerCommandBuilder(basic_docker_settings)

        base_env = {"BASE_VAR": "base_value", "COMMON_VAR": "base_common"}
        cli_env = ["CLI_VAR=cli_value", "COMMON_VAR=cli_common"]

        result = builder._get_merged_environment(cli_env, base_env)

        assert result["BASE_VAR"] == "base_value"
        assert result["CLI_VAR"] == "cli_value"
        assert (
            result["COMMON_VAR"] == "cli_common"
        )  # CLI override should take precedence

    def test_get_merged_environment_invalid_format(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test environment merging with invalid format."""
        builder = DockerCommandBuilder(basic_docker_settings)

        cli_env = ["INVALID_FORMAT"]

        with pytest.raises(ValueError, match="Invalid environment variable format"):
            builder._get_merged_environment(cli_env)

    def test_get_merged_additional_args(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test additional arguments merging functionality."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Set up settings with additional args
        builder.settings.docker_additional_args = ["--privileged", "--network=host"]

        cli_args = ["--user", "1000:1000"]

        result = builder._get_merged_additional_args(cli_args)

        assert "--privileged" in result
        assert "--network=host" in result
        assert "--user" in result
        assert "1000:1000" in result

    def test_get_volumes_for_directories(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test volume creation for directories."""
        builder = DockerCommandBuilder(basic_docker_settings)

        home_dir = "/host/home"
        workspace_dir = "/host/workspace"

        result = builder._get_volumes_for_directories(home_dir, workspace_dir)

        assert len(result) == 2
        assert f"{home_dir}:/data/home" in result
        assert f"{workspace_dir}:/data/workspace" in result

    def test_get_volumes_for_directories_partial(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test volume creation with only one directory specified."""
        builder = DockerCommandBuilder(basic_docker_settings)

        home_dir = "/host/home"
        workspace_dir = None

        result = builder._get_volumes_for_directories(home_dir, workspace_dir)

        assert len(result) == 1
        assert f"{home_dir}:/data/home" in result

    def test_get_environment_for_directories(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test environment variable creation for directories."""
        builder = DockerCommandBuilder(basic_docker_settings)

        # Set up some base environment
        builder.settings.docker_environment = {"EXISTING_VAR": "existing_value"}

        home_dir = "/host/home"
        workspace_dir = "/host/workspace"

        result = builder._get_environment_for_directories(home_dir, workspace_dir)

        assert result["EXISTING_VAR"] == "existing_value"
        assert result["CLAUDE_HOME"] == "/data/home"
        assert result["CLAUDE_WORKSPACE"] == "/data/workspace"

    def test_from_settings_and_overrides(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test convenience class method for building commands."""
        claude_args = ["--help"]
        overrides = {"docker_image": "override-image:latest"}

        cmd = DockerCommandBuilder.from_settings_and_overrides(
            basic_docker_settings, claude_args, **overrides
        )

        assert isinstance(cmd, list)
        assert "docker" in cmd
        assert "run" in cmd
        assert "override-image:latest" in cmd
        assert "claude" in cmd
        assert "--help" in cmd

    def test_complex_scenario(self, tmp_path: Path) -> None:
        """Test a complex scenario with multiple overrides and configurations."""
        # Set up directories
        home_dir = tmp_path / "home"
        workspace_dir = tmp_path / "workspace"
        data_dir = tmp_path / "data"
        home_dir.mkdir()
        workspace_dir.mkdir()
        data_dir.mkdir()

        # Create configured settings
        settings = DockerSettings(
            docker_image="base-image:v1.0",
            docker_volumes=[f"{data_dir}:/app/data"],
            docker_environment={"BASE_ENV": "base_value"},
            docker_additional_args=["--privileged"],
        )

        builder = DockerCommandBuilder(settings)
        claude_args = ["chat", "--model", "claude-3-sonnet", "Hello world"]

        # Create extra directory for volume test
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()

        cmd = builder.build_command(
            claude_args,
            docker_image="custom-image:v2.0",
            docker_env=["CUSTOM_ENV=custom_value"],
            docker_volume=[f"{extra_dir}:/app/extra"],
            docker_arg=["--memory", "4g"],
            docker_home=str(home_dir),
            docker_workspace=str(workspace_dir),
        )

        # Verify the command structure
        assert "docker" in cmd
        assert "run" in cmd
        assert "--rm" in cmd
        assert "-it" in cmd
        assert "custom-image:v2.0" in cmd
        assert "claude" in cmd
        assert "chat" in cmd
        assert "--model" in cmd
        assert "claude-3-sonnet" in cmd
        assert "Hello world" in cmd

        # Verify volumes
        assert "--volume" in cmd
        # Should have original data volume, custom home/workspace, and extra volume

        # Verify environment variables
        assert "--env" in cmd
        # Should have base env, custom env, and directory envs

        # Verify additional args
        assert "--privileged" in cmd  # From settings
        assert "--memory" in cmd  # From override
        assert "4g" in cmd

    def test_env_var_with_equals_in_value(
        self, basic_docker_settings: DockerSettings
    ) -> None:
        """Test environment variable with equals sign in the value."""
        builder = DockerCommandBuilder(basic_docker_settings)
        claude_args = ["--help"]
        docker_env = ["CONNECTION_STRING=host=localhost;port=5432;db=test"]

        cmd = builder.build_command(claude_args, docker_env=docker_env)

        assert "--env" in cmd
        assert "CONNECTION_STRING=host=localhost;port=5432;db=test" in cmd
