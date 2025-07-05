"""Tests for utility helper functions."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from claude_code_proxy.utils.helper import get_package_dir


@pytest.mark.unit
class TestGetPackageDir:
    """Test get_package_dir function."""

    def test_get_package_dir_with_spec_origin(self) -> None:
        """Test get_package_dir when spec and origin are available."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_spec = Mock()
            mock_spec.origin = "/path/to/claude_code_proxy/__init__.py"
            mock_find_spec.return_value = mock_spec

            result = get_package_dir()

            # Should get parent.parent of the spec origin
            expected = Path(
                "/path/to/claude_code_proxy/__init__.py"
            ).parent.parent.resolve()
            assert result == expected

    def test_get_package_dir_with_spec_no_origin(self) -> None:
        """Test get_package_dir when spec exists but has no origin."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_spec = Mock()
            mock_spec.origin = None
            mock_find_spec.return_value = mock_spec

            result = get_package_dir()

            # Should fall back to helper.py path calculation
            # Use actual file path instead of hardcoded path
            from claude_code_proxy.utils.helper import __file__ as helper_file

            expected = Path(helper_file).parent.parent.parent.resolve()
            assert result == expected

    def test_get_package_dir_with_no_spec(self) -> None:
        """Test get_package_dir when find_spec returns None."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.return_value = None

            result = get_package_dir()

            # Should fall back to helper.py path calculation
            # Use actual file path instead of hardcoded path
            from claude_code_proxy.utils.helper import __file__ as helper_file

            expected = Path(helper_file).parent.parent.parent.resolve()
            assert result == expected

    def test_get_package_dir_with_exception(self) -> None:
        """Test get_package_dir when an exception occurs during import."""
        with patch("importlib.util.find_spec") as mock_find_spec:
            mock_find_spec.side_effect = ImportError("Module not found")

            result = get_package_dir()

            # Should fall back to helper.py path calculation in exception handler
            # Use actual file path instead of hardcoded path
            from claude_code_proxy.utils.helper import __file__ as helper_file

            expected = Path(helper_file).parent.parent.parent.resolve()
            assert result == expected

    def test_get_package_dir_returns_path(self) -> None:
        """Test that get_package_dir returns a Path object."""
        result = get_package_dir()
        assert isinstance(result, Path)
        assert result.exists()  # Should be a valid path
