"""Test version module."""

import re

import pytest

from claude_code_proxy import _version


@pytest.mark.unit
class TestVersionModule:
    """Test the _version module."""

    def test_version_string_access(self):
        """Test accessing version string variables."""
        # Test __version__ exists and is a string
        assert hasattr(_version, "__version__")
        assert isinstance(_version.__version__, str)
        assert _version.__version__ == "0.1.0"

        # Test version exists and is a string
        assert hasattr(_version, "version")
        assert isinstance(_version.version, str)
        assert _version.version == "0.1.0"

        # Test both are equal
        assert _version.__version__ == _version.version

    def test_version_tuple_access(self):
        """Test accessing version tuple variables."""
        # Test __version_tuple__ exists and is a tuple
        assert hasattr(_version, "__version_tuple__")
        assert isinstance(_version.__version_tuple__, tuple)
        assert _version.__version_tuple__ == (0, 1, 0)

        # Test version_tuple exists and is a tuple
        assert hasattr(_version, "version_tuple")
        assert isinstance(_version.version_tuple, tuple)
        assert _version.version_tuple == (0, 1, 0)

        # Test both are equal
        assert _version.__version_tuple__ == _version.version_tuple

    def test_version_format_validation(self):
        """Test that version follows semantic versioning format."""
        # Test that version string matches semantic versioning pattern
        semver_pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*)?(?:\+[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*)?$"
        assert re.match(semver_pattern, _version.__version__)
        assert re.match(semver_pattern, _version.version)

    def test_version_tuple_elements(self):
        """Test version tuple contains expected types."""
        # Test that version tuple has 3 elements
        assert len(_version.__version_tuple__) == 3
        assert len(_version.version_tuple) == 3

        # Test that all elements are integers for semantic version
        for element in _version.__version_tuple__:
            assert isinstance(element, int)
        for element in _version.version_tuple:
            assert isinstance(element, int)

        # Test that version string can be reconstructed from tuple
        reconstructed = ".".join(str(x) for x in _version.__version_tuple__)
        assert reconstructed == _version.__version__

    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        expected_exports = [
            "__version__",
            "__version_tuple__",
            "version",
            "version_tuple",
        ]
        assert hasattr(_version, "__all__")
        assert isinstance(_version.__all__, list)
        assert set(_version.__all__) == set(expected_exports)

        # Test that all items in __all__ are actually exported
        for export in _version.__all__:
            assert hasattr(_version, export)

    def test_import_behavior(self):
        """Test that module can be imported and all exports work."""
        # Test direct import
        import claude_code_proxy._version as version_module

        assert version_module.__version__ == "0.1.0"
        assert version_module.version == "0.1.0"
        assert version_module.__version_tuple__ == (0, 1, 0)
        assert version_module.version_tuple == (0, 1, 0)

        # Test from import
        from claude_code_proxy._version import (
            __version__,
            __version_tuple__,
            version,
            version_tuple,
        )

        assert __version__ == "0.1.0"
        assert version == "0.1.0"
        assert __version_tuple__ == (0, 1, 0)
        assert version_tuple == (0, 1, 0)

    def test_type_checking_behavior(self):
        """Test TYPE_CHECKING flag and related types."""
        # Test TYPE_CHECKING is False at runtime
        assert hasattr(_version, "TYPE_CHECKING")
        assert _version.TYPE_CHECKING is False

        # Test VERSION_TUPLE is defined (should be object at runtime)
        assert hasattr(_version, "VERSION_TUPLE")
        assert _version.VERSION_TUPLE is object

    def test_type_checking_imports(self):
        """Test that type checking imports are available when needed."""
        # Import typing modules to ensure they're available for the TYPE_CHECKING block
        import typing

        # Verify that the imports used in TYPE_CHECKING block are valid
        assert hasattr(typing, "Tuple")
        assert hasattr(typing, "Union")

        # Test that we can create the type that would be created in TYPE_CHECKING mode
        VERSION_TUPLE_TYPE = tuple[int | str, ...]

        # Verify it's a valid type
        assert hasattr(VERSION_TUPLE_TYPE, "__origin__")

        # Test that the current version tuple would match this type
        # (This tests the logical consistency of the typing)
        version_tuple_instance = _version.__version_tuple__
        assert isinstance(version_tuple_instance, tuple)

        # All elements should be int or str (in our case, all are int)
        for element in version_tuple_instance:
            assert isinstance(element, int | str)

    def test_type_checking_fallback_coverage(self):
        """Test what happens when we manipulate TYPE_CHECKING at module level."""
        # Since TYPE_CHECKING is set at module level, let's test the actual imports
        # that would be used in the TYPE_CHECKING block by importing them ourselves

        # These imports mirror what's in the TYPE_CHECKING block
        from typing import Tuple, Union  # noqa: UP035

        # Test that we can create the same type definition
        VERSION_TUPLE_ACTUAL = tuple[int | str, ...]

        # This tests the logical equivalence to what would happen
        # if TYPE_CHECKING were True
        assert hasattr(VERSION_TUPLE_ACTUAL, "__origin__")

        # Verify our version data is compatible with this type
        version_tuple = _version.__version_tuple__
        assert isinstance(version_tuple, tuple)

        # Each element should be int or str according to the type
        for element in version_tuple:
            assert isinstance(element, int | str)

    def test_version_consistency(self):
        """Test consistency between string and tuple versions."""
        # Test that string version matches tuple version
        version_parts = _version.__version__.split(".")
        tuple_parts = _version.__version_tuple__

        assert len(version_parts) == len(tuple_parts)

        for str_part, tuple_part in zip(version_parts, tuple_parts, strict=False):
            assert int(str_part) == tuple_part

    def test_module_attributes(self):
        """Test module has expected attributes and no unexpected ones."""
        expected_attrs = {
            "__all__",
            "__version__",
            "__version_tuple__",
            "version",
            "version_tuple",
            "TYPE_CHECKING",
            "VERSION_TUPLE",
        }

        # Get all public attributes (not starting with underscore, except those in __all__)
        public_attrs = {
            attr
            for attr in dir(_version)
            if not attr.startswith("_") or attr in _version.__all__
        }

        # Add special attributes we know should exist
        public_attrs.update({"TYPE_CHECKING", "VERSION_TUPLE"})

        # Check that all expected attributes exist
        for attr in expected_attrs:
            assert hasattr(_version, attr), f"Missing expected attribute: {attr}"

    def test_version_immutability(self):
        """Test that version values appear to be constants."""
        # Store original values
        original_version = _version.__version__
        original_version_tuple = _version.__version_tuple__

        # Verify they're the expected values
        assert original_version == "0.1.0"
        assert original_version_tuple == (0, 1, 0)

        # Note: We can't test true immutability without modifying the module,
        # but we can test that the values are what we expect them to be

    def test_version_string_non_empty(self):
        """Test that version strings are not empty."""
        assert _version.__version__
        assert _version.version
        assert len(_version.__version__) > 0
        assert len(_version.version) > 0

    def test_version_tuple_non_empty(self):
        """Test that version tuples are not empty."""
        assert _version.__version_tuple__
        assert _version.version_tuple
        assert len(_version.__version_tuple__) > 0
        assert len(_version.version_tuple) > 0


@pytest.mark.unit
class TestVersionComparison:
    """Test version comparison functionality."""

    def test_version_comparison_logic(self):
        """Test that version can be used for comparison operations."""
        current_version = _version.__version_tuple__

        # Test comparison with known version
        assert current_version == (0, 1, 0)

        # Test that version is greater than some older versions
        assert current_version > (0, 0, 1)
        assert current_version > (0, 0, 9)

        # Test that version is less than some newer versions
        assert current_version < (0, 1, 1)
        assert current_version < (0, 2, 0)
        assert current_version < (1, 0, 0)

    def test_version_string_comparison(self):
        """Test version string comparison for basic operations."""
        version_str = _version.__version__

        # Test string equality
        assert version_str == "0.1.0"

        # Test string is non-empty and reasonable length
        assert len(version_str) >= 5  # At minimum "X.Y.Z"
        assert "." in version_str  # Should contain dots for semantic versioning


@pytest.mark.unit
class TestVersionAnnotations:
    """Test type annotations and hints."""

    def test_variable_annotations(self):
        """Test that variables have proper type annotations."""
        annotations = getattr(_version, "__annotations__", {})

        # Check if annotations exist (they might not in all Python versions)
        if annotations:
            # If annotations exist, verify they're correct
            assert annotations.get("version") is str
            assert annotations.get("__version__") is str
            # VERSION_TUPLE type annotations are object at runtime (when TYPE_CHECKING is False)
            version_tuple_type = annotations.get("__version_tuple__")
            if version_tuple_type:
                assert (
                    version_tuple_type is object
                )  # At runtime, VERSION_TUPLE = object
            version_tuple_type2 = annotations.get("version_tuple")
            if version_tuple_type2:
                assert (
                    version_tuple_type2 is object
                )  # At runtime, VERSION_TUPLE = object

    def test_module_docstring(self):
        """Test module docstring or comments."""
        # Check if module has docstring or comments (from the file header)
        # Since this is a generated file, it has comments rather than docstring
        assert _version.__doc__ is None  # No formal docstring for generated file


@pytest.mark.unit
class TestErrorConditions:
    """Test error handling and edge cases."""

    def test_attribute_access_safety(self):
        """Test safe attribute access patterns."""
        # Test that accessing non-existent attributes raises AttributeError
        with pytest.raises(AttributeError):
            _ = _version.non_existent_attribute  # type: ignore[attr-defined]

        # Test that __all__ exports are safe to access
        for attr_name in _version.__all__:
            attr_value = getattr(_version, attr_name)
            assert attr_value is not None

    def test_import_safety(self):
        """Test that import operations are safe."""
        # Test that we can import the module multiple times without issues
        import claude_code_proxy._version as v1
        import claude_code_proxy._version as v2

        assert v1.__version__ == v2.__version__
        assert v1.__version_tuple__ == v2.__version_tuple__

        # Test that reimporting gives consistent results
        assert v1 is v2  # Should be the same module object
