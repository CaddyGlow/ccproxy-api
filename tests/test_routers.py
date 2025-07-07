"""Tests for claude_code_proxy.routers module."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestRoutersInit:
    """Tests for claude_code_proxy.routers.__init__ module."""

    def test_module_docstring(self) -> None:
        """Test that the module has the expected docstring."""
        # Read the __init__.py file directly to test its contents
        init_path = (
            Path(__file__).parent.parent
            / "claude_code_proxy"
            / "routers"
            / "__init__.py"
        )
        content = init_path.read_text()

        # Verify the module docstring exists
        assert '"""Router modules for the Claude Proxy API."""' in content

    def test_import_statement_exists(self) -> None:
        """Test that the import statement for chat router exists."""
        # Read the __init__.py file directly to test its contents
        init_path = (
            Path(__file__).parent.parent
            / "claude_code_proxy"
            / "routers"
            / "__init__.py"
        )
        content = init_path.read_text()

        # Verify the import statement exists (line 3)
        assert "from .anthropic import router as anthropic_router" in content

    def test_all_definition(self) -> None:
        """Test that __all__ is defined with the expected content."""
        # Read the __init__.py file directly to test its contents
        init_path = (
            Path(__file__).parent.parent
            / "claude_code_proxy"
            / "routers"
            / "__init__.py"
        )
        content = init_path.read_text()

        # Verify __all__ is defined (line 6)
        assert '__all__ = ["anthropic_router", "openai_router"]' in content

    def test_module_import_with_mocked_routers(self) -> None:
        """Test that the module can be imported when routers are mocked."""
        # Mock both router modules to avoid FastAPI initialization issues
        mock_anthropic_router = MagicMock()
        mock_anthropic_router.routes = []
        mock_anthropic_router.prefix = "/v1"

        mock_openai_router = MagicMock()
        mock_openai_router.routes = []
        mock_openai_router.prefix = "/openai/v1"

        with patch.dict(
            "sys.modules",
            {
                "claude_code_proxy.routers.anthropic": MagicMock(
                    router=mock_anthropic_router
                ),
                "claude_code_proxy.routers.openai": MagicMock(
                    router=mock_openai_router
                ),
            },
        ):
            # Now we can safely import the routers module
            spec = importlib.util.spec_from_file_location(
                "claude_code_proxy.routers",
                Path(__file__).parent.parent
                / "claude_code_proxy"
                / "routers"
                / "__init__.py",
            )
            assert spec is not None
            module = importlib.util.module_from_spec(spec)

            # Execute the module with mocked dependencies
            assert spec.loader is not None
            spec.loader.exec_module(module)

            # Test that __all__ is correctly defined
            assert hasattr(module, "__all__")
            assert module.__all__ == ["anthropic_router", "openai_router"]

            # Test that both routers are available
            assert hasattr(module, "anthropic_router")
            assert hasattr(module, "openai_router")
            assert module.anthropic_router is mock_anthropic_router
            assert module.openai_router is mock_openai_router

    def test_module_attributes_structure(self) -> None:
        """Test the structure of the module without importing dependencies."""
        # This tests the static structure of the module
        init_path = (
            Path(__file__).parent.parent
            / "claude_code_proxy"
            / "routers"
            / "__init__.py"
        )
        lines = init_path.read_text().splitlines()

        # Test specific lines to ensure coverage of lines 3-6
        assert len(lines) >= 6

        # Line 1: docstring
        assert lines[0].startswith('"""')

        # Line 3: import statement (after blank line 2)
        assert "from .anthropic import router as anthropic_router" in lines[2]

        # Line 7: __all__ definition (after blank lines)
        assert '__all__ = ["anthropic_router", "openai_router"]' in lines[6]
