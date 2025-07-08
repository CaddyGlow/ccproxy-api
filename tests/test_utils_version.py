"""Unit tests for ccproxy.utils.version module."""

import pytest

from ccproxy.utils.version import (
    format_version,
    get_next_major_version,
    get_next_minor_version,
    parse_version,
)


@pytest.mark.unit
class TestParseVersion:
    """Test parse_version function."""

    def test_parse_simple_version(self):
        """Test parsing simple semantic version."""
        major, minor, patch, suffix = parse_version("1.2.3")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == ""

    def test_parse_version_with_dev_suffix(self):
        """Test parsing version with .dev suffix."""
        major, minor, patch, suffix = parse_version("1.2.3.dev59")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == "dev"

    def test_parse_setuptools_scm_dev_version(self):
        """Test parsing setuptools-scm development version."""
        major, minor, patch, suffix = parse_version("1.2.3.dev59+g1624e1e.d19800101")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == "dev"

    def test_parse_dev_version_without_patch(self):
        """Test parsing development version without patch number."""
        major, minor, patch, suffix = parse_version("0.1.dev59+g1624e1e.d19800101")
        assert major == 0
        assert minor == 1
        assert patch == 0
        assert suffix == "dev"

    def test_parse_two_part_version(self):
        """Test parsing two-part version (missing patch)."""
        major, minor, patch, suffix = parse_version("1.2")
        assert major == 1
        assert minor == 2
        assert patch == 0
        assert suffix == ""

    def test_parse_single_part_version(self):
        """Test parsing single-part version (missing minor and patch)."""
        major, minor, patch, suffix = parse_version("1")
        assert major == 1
        assert minor == 0
        assert patch == 0
        assert suffix == ""

    def test_parse_zero_version(self):
        """Test parsing zero version."""
        major, minor, patch, suffix = parse_version("0.0.0")
        assert major == 0
        assert minor == 0
        assert patch == 0
        assert suffix == ""

    def test_parse_dev_version_with_patch(self):
        """Test parsing development version with patch number."""
        major, minor, patch, suffix = parse_version("1.2.3.dev59")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == "dev"

    def test_parse_complex_setuptools_scm_version(self):
        """Test parsing complex setuptools-scm version."""
        major, minor, patch, suffix = parse_version("2.1.0.dev123+g456789a.d20230101")
        assert major == 2
        assert minor == 1
        assert patch == 0
        assert suffix == "dev"

    def test_parse_version_with_leading_zeros(self):
        """Test parsing version with leading zeros."""
        major, minor, patch, suffix = parse_version("01.02.03")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == ""

    def test_parse_large_version_numbers(self):
        """Test parsing large version numbers."""
        major, minor, patch, suffix = parse_version("999.888.777")
        assert major == 999
        assert minor == 888
        assert patch == 777
        assert suffix == ""


@pytest.mark.unit
class TestFormatVersion:
    """Test format_version function."""

    def test_format_major_level(self):
        """Test formatting at major level."""
        result = format_version("1.2.3", "major")
        assert result == "1"

    def test_format_minor_level(self):
        """Test formatting at minor level."""
        result = format_version("1.2.3", "minor")
        assert result == "1.2"

    def test_format_patch_level(self):
        """Test formatting at patch level."""
        result = format_version("1.2.3", "patch")
        assert result == "1.2.3"

    def test_format_full_level(self):
        """Test formatting at full level."""
        result = format_version("1.2.3", "full")
        assert result == "1.2.3"

    def test_format_docker_level(self):
        """Test formatting at docker level."""
        result = format_version("1.2.3", "docker")
        assert result == "1.2.3"

    def test_format_npm_level(self):
        """Test formatting at npm level."""
        result = format_version("1.2.3", "npm")
        assert result == "1.2.3"

    def test_format_python_level(self):
        """Test formatting at python level."""
        result = format_version("1.2.3", "python")
        assert result == "1.2.3"

    def test_format_patch_level_with_dev_suffix(self):
        """Test formatting patch level with .dev suffix."""
        result = format_version("1.2.3.dev59", "patch")
        assert result == "1.2.3-dev"

    def test_format_full_level_with_dev_suffix(self):
        """Test formatting full level with .dev suffix."""
        result = format_version("1.2.3.dev59", "full")
        assert result == "1.2.3-dev"

    def test_format_docker_level_with_dev_suffix(self):
        """Test formatting docker level with .dev suffix."""
        result = format_version("1.2.3.dev59", "docker")
        assert result == "1.2.3-dev"

    def test_format_npm_level_with_dev_suffix(self):
        """Test formatting npm level with .dev suffix."""
        result = format_version("1.2.3.dev59", "npm")
        assert result == "1.2.3-dev.0"

    def test_format_python_level_with_dev_suffix(self):
        """Test formatting python level with .dev suffix."""
        result = format_version("1.2.3.dev59", "python")
        assert result == "1.2.3.dev0"

    def test_format_setuptools_scm_version(self):
        """Test formatting setuptools-scm version."""
        result = format_version("1.2.3.dev59+g1624e1e.d19800101", "patch")
        assert result == "1.2.3-dev"

    def test_format_dev_version_without_patch(self):
        """Test formatting development version without patch."""
        result = format_version("0.1.dev59+g1624e1e.d19800101", "patch")
        assert result == "0.1.0-dev"

    def test_format_two_part_version(self):
        """Test formatting two-part version."""
        result = format_version("1.2", "patch")
        assert result == "1.2.0"

    def test_format_single_part_version(self):
        """Test formatting single-part version."""
        result = format_version("1", "patch")
        assert result == "1.0.0"

    def test_format_zero_version(self):
        """Test formatting zero version."""
        result = format_version("0.0.0", "patch")
        assert result == "0.0.0"

    def test_format_major_level_with_dev_suffix(self):
        """Test formatting major level ignores .dev suffix."""
        result = format_version("1.2.3.dev59", "major")
        assert result == "1"

    def test_format_minor_level_with_dev_suffix(self):
        """Test formatting minor level ignores .dev suffix."""
        result = format_version("1.2.3.dev59", "minor")
        assert result == "1.2"

    def test_format_npm_level_complex_suffix(self):
        """Test formatting npm level with complex suffix."""
        result = format_version("1.2.3.dev59+g1624e1e.d19800101", "npm")
        assert result == "1.2.3-dev.0"

    def test_format_python_level_complex_suffix(self):
        """Test formatting python level with complex suffix."""
        result = format_version("1.2.3.dev59+g1624e1e.d19800101", "python")
        assert result == "1.2.3.dev0"

    def test_format_invalid_level(self):
        """Test formatting with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown version level: invalid"):
            format_version("1.2.3", "invalid")

    def test_format_empty_level(self):
        """Test formatting with empty level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown version level: "):
            format_version("1.2.3", "")

    def test_format_none_level(self):
        """Test formatting with None level raises ValueError."""
        with pytest.raises(ValueError, match="Unknown version level: None"):
            format_version("1.2.3", "None")


@pytest.mark.unit
class TestVersionEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_empty_string(self):
        """Test parsing empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("")

    def test_parse_invalid_version_format(self):
        """Test parsing invalid version format raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("not.a.version")

    def test_parse_negative_version_numbers(self):
        """Test parsing negative version numbers works."""
        major, minor, patch, suffix = parse_version("-1.2.3")
        assert major == -1
        assert minor == 2
        assert patch == 3
        assert suffix == ""

    def test_parse_non_numeric_version_parts(self):
        """Test parsing non-numeric version parts raises ValueError."""
        with pytest.raises(ValueError):
            parse_version("a.b.c")

    def test_format_with_invalid_version(self):
        """Test formatting with invalid version raises ValueError."""
        with pytest.raises(ValueError):
            format_version("invalid", "patch")

    def test_parse_version_with_spaces(self):
        """Test parsing version with spaces works."""
        major, minor, patch, suffix = parse_version("1.2.3 ")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == ""

    def test_parse_version_with_special_characters(self):
        """Test parsing version with special characters."""
        # This should work because setuptools-scm uses these patterns
        major, minor, patch, suffix = parse_version("1.2.3.dev59+g1624e1e.d19800101")
        assert major == 1
        assert minor == 2
        assert patch == 3
        assert suffix == "dev"

    def test_format_all_levels_consistency(self):
        """Test that all format levels work consistently."""
        version = "1.2.3.dev59"
        levels = ["major", "minor", "patch", "full", "docker", "npm", "python"]

        for level in levels:
            result = format_version(version, level)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_format_preserves_version_semantics(self):
        """Test that formatting preserves version semantics."""
        version = "1.2.3"

        # Major should be subset of minor
        major = format_version(version, "major")
        minor = format_version(version, "minor")
        assert minor.startswith(major)

        # Minor should be subset of patch
        patch = format_version(version, "patch")
        assert patch.startswith(minor)

    def test_round_trip_consistency(self):
        """Test that parse and format operations are consistent for non-dev versions."""
        # Note: Perfect round-trip consistency is not possible for dev versions
        # because format_version outputs "-dev" but parse_version only accepts ".dev"
        original_versions = [
            "1.2.3",
            "0.1.0",
            "10.20.30",
        ]

        for version in original_versions:
            major, minor, patch, suffix = parse_version(version)

            # Test that we can format back to a reasonable version
            formatted = format_version(version, "patch")

            # Parse the formatted version
            f_major, f_minor, f_patch, f_suffix = parse_version(formatted)

            # Should have same components
            assert f_major == major
            assert f_minor == minor
            assert f_patch == patch
            assert f_suffix == suffix

    def test_dev_version_parsing_and_formatting(self):
        """Test that dev version parsing and formatting work correctly."""
        dev_versions = [
            "1.2.3.dev59",
            "0.1.dev59+g1624e1e.d19800101",
            "1.2.3.dev59+g1624e1e.d19800101",
        ]

        for version in dev_versions:
            major, minor, patch, suffix = parse_version(version)

            # Test that we can format to different levels
            formatted_patch = format_version(version, "patch")
            formatted_docker = format_version(version, "docker")
            formatted_npm = format_version(version, "npm")
            formatted_python = format_version(version, "python")

            # All should be valid strings
            assert isinstance(formatted_patch, str)
            assert isinstance(formatted_docker, str)
            assert isinstance(formatted_npm, str)
            assert isinstance(formatted_python, str)

            # All should contain the dev suffix in some form
            assert "dev" in formatted_patch
            assert "dev" in formatted_docker
            assert "dev" in formatted_npm
            assert "dev" in formatted_python


@pytest.mark.unit
class TestGetNextMinorVersion:
    """Test get_next_minor_version function."""

    def test_simple_version(self):
        """Test getting next minor version for simple version."""
        assert get_next_minor_version("1.2.3") == "1.3.0"

    def test_zero_version(self):
        """Test getting next minor version for zero version."""
        assert get_next_minor_version("0.0.0") == "0.1.0"

    def test_dev_version(self):
        """Test getting next minor version for dev version."""
        assert get_next_minor_version("1.2.3.dev59") == "1.3.0"

    def test_setuptools_scm_version(self):
        """Test getting next minor version for setuptools-scm version."""
        assert get_next_minor_version("1.2.3.dev59+g1624e1e.d19800101") == "1.3.0"

    def test_dev_version_without_patch(self):
        """Test getting next minor version for dev version without patch."""
        assert get_next_minor_version("0.1.dev59+g1624e1e") == "0.2.0"

    def test_large_minor_version(self):
        """Test getting next minor version for large minor number."""
        assert get_next_minor_version("1.999.0") == "1.1000.0"

    def test_two_part_version(self):
        """Test getting next minor version for two-part version."""
        assert get_next_minor_version("1.2") == "1.3.0"

    def test_single_part_version(self):
        """Test getting next minor version for single-part version."""
        assert get_next_minor_version("1") == "1.1.0"

    def test_version_with_high_patch(self):
        """Test getting next minor version resets patch to 0."""
        assert get_next_minor_version("1.2.999") == "1.3.0"

    def test_multiple_increments(self):
        """Test multiple minor version increments."""
        version = "1.0.0"
        version = get_next_minor_version(version)
        assert version == "1.1.0"
        version = get_next_minor_version(version)
        assert version == "1.2.0"
        version = get_next_minor_version(version)
        assert version == "1.3.0"


@pytest.mark.unit
class TestGetNextMajorVersion:
    """Test get_next_major_version function."""

    def test_simple_version(self):
        """Test getting next major version for simple version."""
        assert get_next_major_version("1.2.3") == "2.0.0"

    def test_zero_version(self):
        """Test getting next major version for zero version."""
        assert get_next_major_version("0.0.0") == "1.0.0"

    def test_dev_version(self):
        """Test getting next major version for dev version."""
        assert get_next_major_version("1.2.3.dev59") == "2.0.0"

    def test_setuptools_scm_version(self):
        """Test getting next major version for setuptools-scm version."""
        assert get_next_major_version("1.2.3.dev59+g1624e1e.d19800101") == "2.0.0"

    def test_dev_version_without_patch(self):
        """Test getting next major version for dev version without patch."""
        assert get_next_major_version("0.1.dev59+g1624e1e") == "1.0.0"

    def test_large_major_version(self):
        """Test getting next major version for large major number."""
        assert get_next_major_version("999.888.777") == "1000.0.0"

    def test_two_part_version(self):
        """Test getting next major version for two-part version."""
        assert get_next_major_version("1.2") == "2.0.0"

    def test_single_part_version(self):
        """Test getting next major version for single-part version."""
        assert get_next_major_version("1") == "2.0.0"

    def test_version_with_high_minor_patch(self):
        """Test getting next major version resets minor and patch to 0."""
        assert get_next_major_version("1.999.888") == "2.0.0"

    def test_multiple_increments(self):
        """Test multiple major version increments."""
        version = "0.1.0"
        version = get_next_major_version(version)
        assert version == "1.0.0"
        version = get_next_major_version(version)
        assert version == "2.0.0"
        version = get_next_major_version(version)
        assert version == "3.0.0"
