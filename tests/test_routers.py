"""Tests for ccproxy.routers module."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestRoutersInit:
    """Tests for ccproxy.routers.__init__ module."""

    def test_module_docstring(self) -> None:
        """Test that the module has the expected docstring."""
        # Read the __init__.py file directly to test its contents
        init_path = Path(__file__).parent.parent / "ccproxy" / "routers" / "__init__.py"
        content = init_path.read_text()

        # Verify the module docstring exists
        assert '"""Router modules for the Claude Proxy API."""' in content

    def test_import_statement_exists(self) -> None:
        """Test that the import statement for chat router exists."""
        # Read the __init__.py file directly to test its contents
        init_path = Path(__file__).parent.parent / "ccproxy" / "routers" / "__init__.py"
        content = init_path.read_text()

        # Verify the import statement exists (line 3)
        assert "from .claudecode.anthropic import router as anthropic_router" in content

    def test_all_definition(self) -> None:
        """Test that __all__ is defined with the expected content."""
        # Read the __init__.py file directly to test its contents
        init_path = Path(__file__).parent.parent / "ccproxy" / "routers" / "__init__.py"
        content = init_path.read_text()

        # Verify __all__ is defined (line 9-14)
        assert '"anthropic_router"' in content
        assert '"openai_router"' in content
        assert '"oauth_router"' in content
        assert '"metrics_router"' in content
        assert '"create_reverse_proxy_router"' in content

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
                "ccproxy.routers.claudecode.anthropic": MagicMock(
                    router=mock_anthropic_router
                ),
                "ccproxy.routers.claudecode.openai": MagicMock(
                    router=mock_openai_router
                ),
                "ccproxy.routers.oauth": MagicMock(router=MagicMock()),
                "ccproxy.routers.metrics": MagicMock(router=MagicMock()),
                "ccproxy.routers.reverse_proxy_factory": MagicMock(
                    create_reverse_proxy_router=MagicMock()
                ),
            },
        ):
            # Now we can safely import the routers module
            spec = importlib.util.spec_from_file_location(
                "ccproxy.routers",
                Path(__file__).parent.parent / "ccproxy" / "routers" / "__init__.py",
            )
            assert spec is not None
            module = importlib.util.module_from_spec(spec)

            # Execute the module with mocked dependencies
            assert spec.loader is not None
            spec.loader.exec_module(module)

            # Test that __all__ is correctly defined
            assert hasattr(module, "__all__")
            expected_all = [
                "anthropic_router",
                "openai_router",
                "oauth_router",
                "metrics_router",
                "create_reverse_proxy_router",
            ]
            assert module.__all__ == expected_all

            # Test that all expected exports are available
            assert hasattr(module, "anthropic_router")
            assert hasattr(module, "openai_router")
            assert hasattr(module, "oauth_router")
            assert hasattr(module, "metrics_router")
            assert hasattr(module, "create_reverse_proxy_router")
            assert module.anthropic_router is mock_anthropic_router
            assert module.openai_router is mock_openai_router

    def test_module_attributes_structure(self) -> None:
        """Test the structure of the module without importing dependencies."""
        # This tests the static structure of the module
        init_path = Path(__file__).parent.parent / "ccproxy" / "routers" / "__init__.py"
        lines = init_path.read_text().splitlines()

        # Test specific lines to ensure coverage of lines 3-6
        assert len(lines) >= 6

        # Line 1: docstring
        assert lines[0].startswith('"""')

        # Line 3: import statement (after blank line 2)
        assert (
            "from .claudecode.anthropic import router as anthropic_router" in lines[2]
        )

        # Check __all__ definition is present (not exact line match due to formatting)
        content = "\n".join(lines)
        assert '"anthropic_router"' in content
        assert '"openai_router"' in content
        assert '"metrics_router"' in content
