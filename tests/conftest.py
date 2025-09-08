"""Shared test fixtures and configuration for ccproxy tests.

This module provides minimal, focused fixtures for testing the ccproxy application.
All fixtures have proper type hints and are designed to work with real components
while mocking only external services.
"""

import json
import os
import time
from collections.abc import Generator

# Override settings for testing
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ccproxy.api.app import create_app
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.core.request_context import RequestContext


if TYPE_CHECKING:
    from tests.factories import FastAPIAppFactory, FastAPIClientFactory
from ccproxy.config.core import ServerSettings
from ccproxy.config.security import SecuritySettings
from ccproxy.config.settings import Settings


# Global fixture for task manager (needed by many async tests)
@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Start and stop the global task manager for each test.

    This fixture ensures the AsyncTaskManager is properly started before
    tests that use managed tasks (like PermissionService, scheduler, etc.)
    and properly cleaned up afterwards.
    """
    await start_task_manager()
    yield
    await stop_task_manager()


# Plugin fixtures are declared in root-level conftest.py


@lru_cache
def get_test_settings(test_settings: Settings) -> Settings:
    """Get test settings - overrides the default settings provider."""
    return test_settings


# Test data directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def isolated_environment(tmp_path: Path) -> Generator[Path, None, None]:
    """Create isolated test environment with XDG directories and working directory.

    Returns an isolated temporary directory and sets environment variables
    to ensure complete test isolation for file system operations.

    Sets up:
    - HOME to point to temporary directory
    - XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME to subdirectories
    - Changes working directory to the temporary directory (for Claude SDK)
    - Creates the necessary directory structure
    """
    # Set up XDG base directories within the temp path
    home_dir = tmp_path / "home"
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"

    # Create directories
    home_dir.mkdir()
    config_dir.mkdir()
    data_dir.mkdir()
    cache_dir.mkdir()

    # Store original environment variables and working directory
    original_env = {
        "HOME": os.environ.get("HOME"),
        "XDG_CONFIG_HOME": os.environ.get("XDG_CONFIG_HOME"),
        "XDG_DATA_HOME": os.environ.get("XDG_DATA_HOME"),
        "XDG_CACHE_HOME": os.environ.get("XDG_CACHE_HOME"),
    }
    original_cwd = Path.cwd()

    try:
        # Change to isolated working directory (important for Claude SDK)
        os.chdir(tmp_path)

        # Set isolated environment variables
        with patch.dict(
            os.environ,
            {
                "HOME": str(home_dir),
                "XDG_CONFIG_HOME": str(config_dir),
                "XDG_DATA_HOME": str(data_dir),
                "XDG_CACHE_HOME": str(cache_dir),
            },
        ):
            yield tmp_path
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Environment variables are automatically restored by patch.dict context manager


@pytest.fixture
def claude_sdk_environment(isolated_environment: Path) -> Path:
    """Create Claude SDK-specific test environment with MCP configuration.

    This fixture extends isolated_environment to create a proper Claude SDK
    test environment with:
    - Basic MCP configuration file
    - Claude configuration directory
    - Proper working directory setup
    """
    # create claude config directory structure
    claude_config_dir = isolated_environment / ".claude"
    claude_config_dir.mkdir(exist_ok=True)

    # Create a minimal MCP configuration to prevent errors
    mcp_config = {"mcpServers": {"test": {"command": "echo", "args": ["test"]}}}

    mcp_config_file = isolated_environment / ".mcp.json"
    mcp_config_file.write_text(json.dumps(mcp_config))

    return isolated_environment


@pytest.fixture
def test_settings(isolated_environment: Path) -> Settings:
    """Create isolated test settings with temp directories.

    Returns a Settings instance configured for testing with:
    - Temporary config and cache directories using isolated environment
    - Observability endpoints enabled for testing
    - No authentication by default
    - Test environment enabled
    """
    return Settings(
        server=ServerSettings(log_level="WARNING"),
        security=SecuritySettings(auth_token=None),  # No auth by default
        plugins={
            "duckdb_storage": {
                "enabled": True,
                "database_path": str(isolated_environment / "test_metrics.duckdb"),
                "register_app_state_alias": True,
            },
            "analytics": {"enabled": True},
        },
    )


@pytest.fixture
def auth_settings(isolated_environment: Path) -> Settings:
    """Create test settings with authentication enabled.

    Returns a Settings instance configured for testing with authentication:
    - Temporary config and cache directories using isolated environment
    - Authentication token configured for testing
    - Observability endpoints enabled for testing
    """
    return Settings(
        server=ServerSettings(log_level="WARNING"),
        security=SecuritySettings(
            auth_token=SecretStr("test-auth-token-12345")
        ),  # Auth enabled
        plugins={
            "duckdb_storage": {
                "enabled": True,
                "database_path": str(isolated_environment / "test_metrics.duckdb"),
                "register_app_state_alias": True,
            },
            "analytics": {"enabled": True},
        },
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """Create test FastAPI application with test settings.

    Returns a configured FastAPI app ready for testing.
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.api.dependencies import get_cached_settings
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    def mock_get_cached_settings_for_test(request: Request):
        return test_settings

    app.dependency_overrides[get_cached_settings] = mock_get_cached_settings_for_test

    return app


@pytest.fixture
def app_with_claude_sdk_environment(
    claude_sdk_environment: Path,
    test_settings: Settings,
    mock_internal_claude_sdk_service: AsyncMock,
) -> FastAPI:
    """Create test FastAPI application with Claude SDK environment and mocked service.

    This fixture provides a properly configured Claude SDK environment with:
    - Isolated working directory
    - MCP configuration files
    - Environment variables set up
    - Mocked Claude service to prevent actual CLI execution
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.api.dependencies import get_cached_settings
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    def mock_get_cached_settings_for_claude_sdk(request: Request):
        return test_settings

    app.dependency_overrides[get_cached_settings] = (
        mock_get_cached_settings_for_claude_sdk
    )

    # NOTE: Plugin-based architecture no longer uses get_cached_claude_service
    # Store mock in app state for compatibility if needed by tests
    app.state.claude_service_mock = mock_internal_claude_sdk_service

    return app


@pytest.fixture
def client_with_claude_sdk_environment(
    app_with_claude_sdk_environment: FastAPI,
) -> TestClient:
    """Create test client with Claude SDK environment setup.

    Returns a TestClient configured with proper Claude SDK environment isolation.
    """
    return TestClient(app_with_claude_sdk_environment)


@pytest.fixture
def claude_responses() -> dict[str, Any]:
    """Load standard Claude API responses from fixtures.

    Returns a dictionary of mock Claude API responses.
    """
    responses_file = FIXTURES_DIR / "responses.json"
    if responses_file.exists():
        response_data = json.loads(responses_file.read_text())
        return response_data  # type: ignore[no-any-return]

    # Default responses if file doesn't exist yet
    return {
        "standard_completion": {
            "id": "msg_01234567890",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        },
        "error_response": {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified",
            },
        },
    }


# Basic auth mode fixtures
@pytest.fixture
def auth_mode_none() -> dict[str, Any]:
    """Auth mode: No authentication required."""
    return {"mode": "none", "requires_token": False}


@pytest.fixture
def auth_mode_bearer_token() -> dict[str, Any]:
    """Auth mode: Bearer token authentication."""
    return {
        "mode": "bearer_token",
        "requires_token": True,
        "test_token": "test-bearer-token-12345",
    }


@pytest.fixture
def auth_mode_configured_token() -> dict[str, Any]:
    """Auth mode: Bearer token with server-configured auth_token."""
    return {
        "mode": "configured_token",
        "requires_token": True,
        "server_token": "server-configured-token-67890",
        "test_token": "server-configured-token-67890",
        "invalid_token": "wrong-token-12345",
    }


# Factory pattern fixtures
@pytest.fixture
def fastapi_app_factory(test_settings: Settings) -> "FastAPIAppFactory":
    """Create FastAPI app factory for flexible test app creation."""
    from tests.factories import FastAPIAppFactory

    return FastAPIAppFactory(default_settings=test_settings)


@pytest.fixture
def fastapi_client_factory(
    fastapi_app_factory: "FastAPIAppFactory",
) -> "FastAPIClientFactory":
    """Create FastAPI client factory for flexible test client creation."""
    from tests.factories import FastAPIClientFactory

    return FastAPIClientFactory(fastapi_app_factory)


# Missing fixtures for API tests compatibility


@pytest.fixture
def client_with_mock_claude(
    test_settings: Settings,
    mock_internal_claude_sdk_service: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with mocked Claude service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_internal_claude_sdk_service,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def client_with_mock_claude_streaming(
    test_settings: Settings,
    mock_internal_claude_sdk_service_streaming: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with mocked Claude streaming service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_internal_claude_sdk_service_streaming,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def client_with_unavailable_claude(
    test_settings: Settings,
    mock_internal_claude_sdk_service_unavailable: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with unavailable Claude service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_internal_claude_sdk_service_unavailable,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Auth headers for bearer token authentication."""
    return {"Authorization": "Bearer test-bearer-token-12345"}


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Basic test client."""
    return TestClient(app)


# Codex-specific fixtures following Claude patterns


@pytest.fixture
def codex_responses() -> dict[str, Any]:
    """Load standard Codex API responses for testing.

    Returns a dictionary of mock Codex API responses.
    """
    return {
        "standard_completion": {
            "id": "codex_01234567890",
            "object": "codex.response",
            "created": 1234567890,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you with coding today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        },
        "error_response": {
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified",
                "code": "invalid_model",
            }
        },
    }


# Test Utilities


def create_test_request_context(request_id: str, **metadata: Any) -> "RequestContext":
    """Create a RequestContext for testing with proper parameters.

    Args:
        request_id: The request ID for the context
        **metadata: Additional metadata to include in the context

    Returns:
        RequestContext: A properly initialized context for testing
    """
    # Create a test logger
    logger = structlog.get_logger(__name__).bind(request_id=request_id)

    # Create context with required parameters
    context = RequestContext(
        request_id=request_id,
        start_time=time.perf_counter(),
        logger=logger,
    )

    # Add any metadata
    if metadata:
        context.add_metadata(**metadata)

    return context


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom settings."""
    # Ensure async tests work properly
    config.option.asyncio_mode = "auto"


# Test directory validation
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark async tests
        if "async" in item.nodeid:
            item.add_marker(pytest.mark.asyncio)

        # Add unit marker to tests not marked as real_api
        if not any(marker.name == "real_api" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
