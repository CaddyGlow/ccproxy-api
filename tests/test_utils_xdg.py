"""Test utils XDG functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ccproxy.utils.xdg import (
    get_ccproxy_config_dir,
    get_claude_cli_config_dir,
    get_claude_docker_home_dir,
    get_xdg_cache_home,
    get_xdg_config_home,
    get_xdg_data_home,
)


class TestXDGConfigHome:
    """Test get_xdg_config_home function."""

    def test_get_xdg_config_home_with_env_var(self):
        """Test get_xdg_config_home when XDG_CONFIG_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_config"
            config_path.mkdir()

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_path)}):
                result = get_xdg_config_home()
                assert result == config_path
                assert isinstance(result, Path)

    def test_get_xdg_config_home_without_env_var(self):
        """Test get_xdg_config_home when XDG_CONFIG_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_config_home()
            expected = Path("/home/testuser") / ".config"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_config_home_empty_env_var(self):
        """Test get_xdg_config_home when XDG_CONFIG_HOME is empty string."""
        with (
            patch.dict(os.environ, {"XDG_CONFIG_HOME": ""}),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_config_home()
            expected = Path("/home/testuser") / ".config"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_config_home_whitespace_env_var(self):
        """Test get_xdg_config_home when XDG_CONFIG_HOME is whitespace."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "   "}):
            result = get_xdg_config_home()
            expected = Path("   ")
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_config_home_relative_path(self):
        """Test get_xdg_config_home with relative path in env var."""
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "./config"}):
            result = get_xdg_config_home()
            expected = Path("./config")
            assert result == expected
            assert isinstance(result, Path)


class TestXDGDataHome:
    """Test get_xdg_data_home function."""

    def test_get_xdg_data_home_with_env_var(self):
        """Test get_xdg_data_home when XDG_DATA_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "custom_data"
            data_path.mkdir()

            with patch.dict(os.environ, {"XDG_DATA_HOME": str(data_path)}):
                result = get_xdg_data_home()
                assert result == data_path
                assert isinstance(result, Path)

    def test_get_xdg_data_home_without_env_var(self):
        """Test get_xdg_data_home when XDG_DATA_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_data_home()
            expected = Path("/home/testuser") / ".local" / "share"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_data_home_empty_env_var(self):
        """Test get_xdg_data_home when XDG_DATA_HOME is empty string."""
        with (
            patch.dict(os.environ, {"XDG_DATA_HOME": ""}),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_data_home()
            expected = Path("/home/testuser") / ".local" / "share"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_data_home_whitespace_env_var(self):
        """Test get_xdg_data_home when XDG_DATA_HOME is whitespace."""
        with patch.dict(os.environ, {"XDG_DATA_HOME": "   "}):
            result = get_xdg_data_home()
            expected = Path("   ")
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_data_home_relative_path(self):
        """Test get_xdg_data_home with relative path in env var."""
        with patch.dict(os.environ, {"XDG_DATA_HOME": "./data"}):
            result = get_xdg_data_home()
            expected = Path("./data")
            assert result == expected
            assert isinstance(result, Path)


class TestXDGCacheHome:
    """Test get_xdg_cache_home function."""

    def test_get_xdg_cache_home_with_env_var(self):
        """Test get_xdg_cache_home when XDG_CACHE_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "custom_cache"
            cache_path.mkdir()

            with patch.dict(os.environ, {"XDG_CACHE_HOME": str(cache_path)}):
                result = get_xdg_cache_home()
                assert result == cache_path
                assert isinstance(result, Path)

    def test_get_xdg_cache_home_without_env_var(self):
        """Test get_xdg_cache_home when XDG_CACHE_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_cache_home()
            expected = Path("/home/testuser") / ".cache"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_cache_home_empty_env_var(self):
        """Test get_xdg_cache_home when XDG_CACHE_HOME is empty string."""
        with (
            patch.dict(os.environ, {"XDG_CACHE_HOME": ""}),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_xdg_cache_home()
            expected = Path("/home/testuser") / ".cache"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_cache_home_whitespace_env_var(self):
        """Test get_xdg_cache_home when XDG_CACHE_HOME is whitespace."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "   "}):
            result = get_xdg_cache_home()
            expected = Path("   ")
            assert result == expected
            assert isinstance(result, Path)

    def test_get_xdg_cache_home_relative_path(self):
        """Test get_xdg_cache_home with relative path in env var."""
        with patch.dict(os.environ, {"XDG_CACHE_HOME": "./cache"}):
            result = get_xdg_cache_home()
            expected = Path("./cache")
            assert result == expected
            assert isinstance(result, Path)


class TestCCProxyConfigDir:
    """Test get_ccproxy_config_dir function."""

    def test_get_ccproxy_config_dir_with_xdg_config_home(self):
        """Test get_ccproxy_config_dir when XDG_CONFIG_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_config"
            config_path.mkdir()

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_path)}):
                result = get_ccproxy_config_dir()
                expected = config_path / "ccproxy"
                assert result == expected
                assert isinstance(result, Path)

    def test_get_ccproxy_config_dir_without_xdg_config_home(self):
        """Test get_ccproxy_config_dir when XDG_CONFIG_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_ccproxy_config_dir()
            expected = Path("/home/testuser") / ".config" / "ccproxy"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_ccproxy_config_dir_dependency_on_get_xdg_config_home(self):
        """Test that get_ccproxy_config_dir properly depends on get_xdg_config_home."""
        with patch("ccproxy.utils.xdg.get_xdg_config_home") as mock_xdg:
            mock_xdg.return_value = Path("/custom/config")
            result = get_ccproxy_config_dir()
            expected = Path("/custom/config") / "ccproxy"
            assert result == expected
            assert isinstance(result, Path)
            mock_xdg.assert_called_once()


class TestClaudeCLIConfigDir:
    """Test get_claude_cli_config_dir function."""

    def test_get_claude_cli_config_dir_with_xdg_config_home(self):
        """Test get_claude_cli_config_dir when XDG_CONFIG_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_config"
            config_path.mkdir()

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_path)}):
                result = get_claude_cli_config_dir()
                expected = config_path / "claude"
                assert result == expected
                assert isinstance(result, Path)

    def test_get_claude_cli_config_dir_without_xdg_config_home(self):
        """Test get_claude_cli_config_dir when XDG_CONFIG_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_claude_cli_config_dir()
            expected = Path("/home/testuser") / ".config" / "claude"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_claude_cli_config_dir_dependency_on_get_xdg_config_home(self):
        """Test that get_claude_cli_config_dir properly depends on get_xdg_config_home."""
        with patch("ccproxy.utils.xdg.get_xdg_config_home") as mock_xdg:
            mock_xdg.return_value = Path("/custom/config")
            result = get_claude_cli_config_dir()
            expected = Path("/custom/config") / "claude"
            assert result == expected
            assert isinstance(result, Path)
            mock_xdg.assert_called_once()


class TestClaudeDockerHomeDir:
    """Test get_claude_docker_home_dir function."""

    def test_get_claude_docker_home_dir_with_xdg_config_home(self):
        """Test get_claude_docker_home_dir when XDG_CONFIG_HOME is set."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom_config"
            config_path.mkdir()

            with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(config_path)}):
                result = get_claude_docker_home_dir()
                expected = config_path / "ccproxy" / "home"
                assert result == expected
                assert isinstance(result, Path)

    def test_get_claude_docker_home_dir_without_xdg_config_home(self):
        """Test get_claude_docker_home_dir when XDG_CONFIG_HOME is not set."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")
            result = get_claude_docker_home_dir()
            expected = Path("/home/testuser") / ".config" / "ccproxy" / "home"
            assert result == expected
            assert isinstance(result, Path)

    def test_get_claude_docker_home_dir_dependency_on_get_ccproxy_config_dir(self):
        """Test that get_claude_docker_home_dir properly depends on get_ccproxy_config_dir."""
        with patch("ccproxy.utils.xdg.get_ccproxy_config_dir") as mock_ccproxy:
            mock_ccproxy.return_value = Path("/custom/ccproxy")
            result = get_claude_docker_home_dir()
            expected = Path("/custom/ccproxy") / "home"
            assert result == expected
            assert isinstance(result, Path)
            mock_ccproxy.assert_called_once()


class TestXDGIntegration:
    """Integration tests for XDG functions."""

    def test_all_functions_return_path_objects(self):
        """Test that all XDG functions return Path objects."""
        functions = [
            get_xdg_config_home,
            get_xdg_data_home,
            get_xdg_cache_home,
            get_ccproxy_config_dir,
            get_claude_cli_config_dir,
            get_claude_docker_home_dir,
        ]

        for func in functions:
            result = func()
            assert isinstance(result, Path), (
                f"{func.__name__} should return Path object"
            )

    def test_hierarchy_consistency(self):
        """Test that the directory hierarchy is consistent."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")

            # Test the hierarchy
            config_home = get_xdg_config_home()
            ccproxy_config = get_ccproxy_config_dir()
            claude_cli_config = get_claude_cli_config_dir()
            claude_docker_home = get_claude_docker_home_dir()

            # Verify hierarchy
            assert ccproxy_config.parent == config_home
            assert claude_cli_config.parent == config_home
            assert claude_docker_home.parent == ccproxy_config

            # Verify specific paths
            assert ccproxy_config.name == "ccproxy"
            assert claude_cli_config.name == "claude"
            assert claude_docker_home.name == "home"

    def test_mixed_environment_variables(self):
        """Test behavior with mixed environment variable states."""
        with (
            patch.dict(
                os.environ,
                {"XDG_CONFIG_HOME": "/custom/config", "XDG_DATA_HOME": ""},
                clear=True,
            ),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.return_value = Path("/home/testuser")

            # CONFIG_HOME should use env var
            config_home = get_xdg_config_home()
            assert config_home == Path("/custom/config")

            # DATA_HOME should fall back to default (empty string)
            data_home = get_xdg_data_home()
            assert data_home == Path("/home/testuser") / ".local" / "share"

            # ccproxy should use custom config
            ccproxy_config = get_ccproxy_config_dir()
            assert ccproxy_config == Path("/custom/config") / "ccproxy"


class TestXDGEdgeCases:
    """Test edge cases and error conditions."""

    def test_path_with_special_characters(self):
        """Test XDG functions with paths containing special characters."""
        special_path = "/tmp/test dir/with spaces & symbols"

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": special_path}):
            result = get_xdg_config_home()
            assert result == Path(special_path)
            assert isinstance(result, Path)

    def test_unicode_path(self):
        """Test XDG functions with Unicode paths."""
        unicode_path = "/tmp/测试目录/配置"

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": unicode_path}):
            result = get_xdg_config_home()
            assert result == Path(unicode_path)
            assert isinstance(result, Path)

    def test_very_long_path(self):
        """Test XDG functions with very long paths."""
        long_path = "/tmp/" + "very_long_directory_name" * 10

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": long_path}):
            result = get_xdg_config_home()
            assert result == Path(long_path)
            assert isinstance(result, Path)

    def test_home_directory_access_error(self):
        """Test behavior when Path.home() raises an exception."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ccproxy.utils.xdg.Path.home") as mock_home,
        ):
            mock_home.side_effect = RuntimeError("Cannot determine home directory")

            with pytest.raises(RuntimeError, match="Cannot determine home directory"):
                get_xdg_config_home()

    def test_none_env_var_behavior(self):
        """Test behavior when environment variable is explicitly None."""
        # This tests the os.environ.get() behavior
        with patch("ccproxy.utils.xdg.os.environ.get") as mock_get:
            mock_get.return_value = None

            with patch("ccproxy.utils.xdg.Path.home") as mock_home:
                mock_home.return_value = Path("/home/testuser")

                result = get_xdg_config_home()
                expected = Path("/home/testuser") / ".config"
                assert result == expected
                mock_get.assert_called_once_with("XDG_CONFIG_HOME")
