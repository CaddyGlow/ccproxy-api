"""Shared test fixtures and configuration for ccproxy tests.

This module provides minimal, focused fixtures for testing the ccproxy application.
All fixtures have proper type hints and are designed to work with real components
while mocking only external services.
"""

import json
import os
import time
from collections.abc import Callable, Generator

# Override settings for testing
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ccproxy.api.app import create_app
from ccproxy.observability.context import RequestContext


if TYPE_CHECKING:
    from tests.factories import FastAPIAppFactory, FastAPIClientFactory
from ccproxy.auth.manager import AuthManager
from ccproxy.config.auth import AuthSettings, CredentialStorageSettings
from ccproxy.config.observability import ObservabilitySettings
from ccproxy.config.security import SecuritySettings
from ccproxy.config.server import ServerSettings
from ccproxy.config.settings import Settings
from ccproxy.docker.adapter import DockerAdapter
from ccproxy.docker.docker_path import DockerPath, DockerPathSet
from ccproxy.docker.models import DockerUserContext
from ccproxy.docker.stream_process import DefaultOutputMiddleware

# Import organized fixture modules
from tests.fixtures.claude_sdk.internal_mocks import (
    mock_internal_claude_sdk_service,
    mock_internal_claude_sdk_service_streaming,
    mock_internal_claude_sdk_service_unavailable,
)
from tests.fixtures.external_apis.anthropic_api import (
    mock_external_anthropic_api,
    mock_external_anthropic_api_streaming,
)
from tests.fixtures.proxy_service.oauth_mocks import mock_external_oauth_endpoints


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================
# These aliases maintain compatibility while the new organized fixtures are adopted

# Backward compatibility aliases for Claude SDK internal mocks
# OLD NAME → NEW NAME (explanation)
mock_claude_service = (
    mock_internal_claude_sdk_service  # Internal dependency injection mock
)
mock_claude_service_streaming = (
    mock_internal_claude_sdk_service_streaming  # Internal streaming mock
)
mock_claude_service_unavailable = (
    mock_internal_claude_sdk_service_unavailable  # Internal unavailable mock
)

# Backward compatibility aliases for external API mocks
# OLD NAME → NEW NAME (explanation)
mock_claude = mock_external_anthropic_api  # External HTTP interception mock
mock_claude_stream = (
    mock_external_anthropic_api_streaming  # External streaming HTTP mock
)
mock_oauth = mock_external_oauth_endpoints  # External OAuth endpoints mock

# =============================================================================


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
    # Create Claude config directory structure
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
        auth=AuthSettings(
            storage=CredentialStorageSettings(
                storage_paths=[isolated_environment / ".claude/"]
            )
        ),
        observability=ObservabilitySettings(
            # Enable all observability endpoints for testing
            metrics_endpoint_enabled=True,
            logs_endpoints_enabled=True,
            logs_collection_enabled=True,
            dashboard_enabled=True,
            log_storage_backend="duckdb",
            duckdb_path=str(isolated_environment / "test_metrics.duckdb"),
        ),
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
        security=SecuritySettings(auth_token="test-auth-token-12345"),  # Auth enabled
        auth=AuthSettings(
            storage=CredentialStorageSettings(
                storage_paths=[isolated_environment / ".claude/"]
            )
        ),
        observability=ObservabilitySettings(
            # Enable all observability endpoints for testing
            metrics_endpoint_enabled=True,
            logs_endpoints_enabled=True,
            logs_collection_enabled=True,
            dashboard_enabled=True,
            log_storage_backend="duckdb",
            duckdb_path=str(isolated_environment / "test_metrics.duckdb"),
        ),
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
    mock_claude_service: AsyncMock,
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
    from ccproxy.api.dependencies import get_cached_claude_service, get_cached_settings
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    def mock_get_cached_settings_for_claude_sdk(request: Request):
        return test_settings

    app.dependency_overrides[get_cached_settings] = (
        mock_get_cached_settings_for_claude_sdk
    )

    # Override the actual dependency being used (get_cached_claude_service)
    def mock_get_cached_claude_service_for_sdk(request: Request) -> AsyncMock:
        return mock_claude_service

    app.dependency_overrides[get_cached_claude_service] = (
        mock_get_cached_claude_service_for_sdk
    )

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


@pytest.fixture
def metrics_storage() -> Any:
    """Create isolated in-memory metrics storage.

    Returns a mock storage instance for testing.
    """
    return None


# =============================================================================
# COMPOSABLE AUTH FIXTURE HIERARCHY
# =============================================================================
# New composable auth fixtures that support all auth modes without skipping


@pytest.fixture
def auth_mode_none() -> dict[str, Any]:
    """Auth mode: No authentication required.

    Returns configuration for testing endpoints without authentication.
    """
    return {
        "mode": "none",
        "requires_token": False,
        "has_configured_token": False,
        "credentials_available": False,
    }


@pytest.fixture
def auth_mode_bearer_token() -> dict[str, Any]:
    """Auth mode: Bearer token authentication without configured server token.

    Returns configuration for testing with bearer tokens when server has no auth_token configured.
    """
    return {
        "mode": "bearer_token",
        "requires_token": True,
        "has_configured_token": False,
        "credentials_available": False,
        "test_token": "test-bearer-token-12345",
    }


@pytest.fixture
def auth_mode_configured_token() -> dict[str, Any]:
    """Auth mode: Bearer token with server-configured auth_token.

    Returns configuration for testing with bearer tokens when server has auth_token configured.
    """
    return {
        "mode": "configured_token",
        "requires_token": True,
        "has_configured_token": True,
        "credentials_available": False,
        "server_token": "server-configured-token-67890",
        "test_token": "server-configured-token-67890",  # Must match server
        "invalid_token": "wrong-token-12345",
    }


@pytest.fixture
def auth_mode_credentials() -> dict[str, Any]:
    """Auth mode: Credentials-based authentication (OAuth flow).

    Returns configuration for testing with Claude SDK credentials.
    """
    return {
        "mode": "credentials",
        "requires_token": False,
        "has_configured_token": False,
        "credentials_available": True,
    }


@pytest.fixture
def auth_mode_credentials_with_fallback() -> dict[str, Any]:
    """Auth mode: Credentials with bearer token fallback.

    Returns configuration for testing both credentials and bearer token support.
    """
    return {
        "mode": "credentials_with_fallback",
        "requires_token": False,
        "has_configured_token": False,
        "credentials_available": True,
        "test_token": "fallback-bearer-token-12345",
    }


# Auth Settings Factories
@pytest.fixture
def auth_settings_factory() -> Callable[[dict[str, Any]], Settings]:
    """Factory for creating auth-specific settings.

    Returns a function that creates Settings based on auth mode configuration.
    """

    def _create_settings(auth_config: dict[str, Any]) -> Settings:
        # Create base test settings
        settings = Settings(
            server=ServerSettings(log_level="WARNING"),
            security=SecuritySettings(auth_token=None),
            auth=AuthSettings(
                storage=CredentialStorageSettings(
                    storage_paths=[Path("/tmp/test/.claude/")]
                )
            ),
        )

        if auth_config.get("has_configured_token"):
            settings.security.auth_token = auth_config["server_token"]
        else:
            settings.security.auth_token = None

        return settings

    return _create_settings


# Auth Headers Generators
@pytest.fixture
def auth_headers_factory() -> Callable[[dict[str, Any]], dict[str, str]]:
    """Factory for creating auth headers based on auth mode.

    Returns a function that creates appropriate headers for each auth mode.
    """

    def _create_headers(auth_config: dict[str, Any]) -> dict[str, str]:
        if not auth_config.get("requires_token"):
            return {}

        token = auth_config.get("test_token")
        if not token:
            return {}

        return {"Authorization": f"Bearer {token}"}

    return _create_headers


@pytest.fixture
def invalid_auth_headers_factory() -> Callable[[dict[str, Any]], dict[str, str]]:
    """Factory for creating invalid auth headers for negative testing.

    Returns a function that creates headers with invalid tokens.
    """

    def _create_invalid_headers(auth_config: dict[str, Any]) -> dict[str, str]:
        if auth_config["mode"] == "configured_token":
            return {"Authorization": f"Bearer {auth_config['invalid_token']}"}
        elif auth_config["mode"] in ["bearer_token", "credentials_with_fallback"]:
            return {"Authorization": "Bearer invalid-token-99999"}
        else:
            return {"Authorization": "Bearer should-fail-12345"}

    return _create_invalid_headers


# Composable App Fixtures
@pytest.fixture
def app_factory(tmp_path: Path) -> Callable[[dict[str, Any]], FastAPI]:
    """Factory for creating FastAPI apps with specific auth configurations.

    Returns a function that creates apps based on auth mode configuration.
    """

    def _create_app(auth_config: dict[str, Any]) -> FastAPI:
        # Create settings based on auth config
        settings = Settings(
            server=ServerSettings(log_level="WARNING"),
            security=SecuritySettings(auth_token=None),
            auth=AuthSettings(
                storage=CredentialStorageSettings(storage_paths=[tmp_path / ".claude/"])
            ),
            observability=ObservabilitySettings(
                # Enable all observability endpoints for testing
                metrics_endpoint_enabled=True,
                logs_endpoints_enabled=True,
                logs_collection_enabled=True,
                dashboard_enabled=True,
                log_storage_backend="duckdb",
                duckdb_path=str(tmp_path / "test_metrics.duckdb"),
            ),
        )
        if auth_config.get("has_configured_token"):
            settings.security.auth_token = auth_config["server_token"]
        else:
            settings.security.auth_token = None

        # Create app with settings
        app = create_app(settings=settings)

        # Override settings dependency for testing
        from ccproxy.api.dependencies import get_cached_settings
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: settings

        def mock_get_cached_settings_for_factory(request: Request):
            return settings

        app.dependency_overrides[get_cached_settings] = (
            mock_get_cached_settings_for_factory
        )

        # Override auth manager if needed
        if auth_config["mode"] != "none":
            from fastapi.security import HTTPAuthorizationCredentials

            from ccproxy.auth.dependencies import (
                _get_auth_manager_with_settings,
                get_auth_manager,
            )

            async def test_auth_manager(
                credentials: HTTPAuthorizationCredentials | None = None,
            ) -> AuthManager:
                return await _get_auth_manager_with_settings(credentials, settings)

            app.dependency_overrides[get_auth_manager] = test_auth_manager

        return app

    return _create_app


@pytest.fixture
def client_factory() -> Callable[[FastAPI], TestClient]:
    """Factory for creating test clients from FastAPI apps.

    Returns a function that creates TestClient instances.
    """

    def _create_client(app: FastAPI) -> TestClient:
        return TestClient(app)

    return _create_client


# Specific Mode Fixtures (for convenience)
@pytest.fixture
def app_no_auth(
    auth_mode_none: dict[str, Any], app_factory: Callable[[dict[str, Any]], FastAPI]
) -> FastAPI:
    """FastAPI app with no authentication required."""
    return app_factory(auth_mode_none)


@pytest.fixture
def app_bearer_auth(
    auth_mode_bearer_token: dict[str, Any],
    app_factory: Callable[[dict[str, Any]], FastAPI],
) -> FastAPI:
    """FastAPI app with bearer token authentication (no configured token)."""
    return app_factory(auth_mode_bearer_token)


@pytest.fixture
def app_configured_auth(
    auth_mode_configured_token: dict[str, Any],
    app_factory: Callable[[dict[str, Any]], FastAPI],
) -> FastAPI:
    """FastAPI app with configured auth token."""
    return app_factory(auth_mode_configured_token)


@pytest.fixture
def app_credentials_auth(
    auth_mode_credentials: dict[str, Any],
    app_factory: Callable[[dict[str, Any]], FastAPI],
) -> FastAPI:
    """FastAPI app with credentials-based authentication."""
    return app_factory(auth_mode_credentials)


@pytest.fixture
def client_no_auth(
    app_no_auth: FastAPI, client_factory: Callable[[FastAPI], TestClient]
) -> TestClient:
    """Test client with no authentication."""
    return client_factory(app_no_auth)


@pytest.fixture
def client_bearer_auth(
    app_bearer_auth: FastAPI, client_factory: Callable[[FastAPI], TestClient]
) -> TestClient:
    """Test client with bearer token authentication."""
    return client_factory(app_bearer_auth)


@pytest.fixture
def client_configured_auth(
    app_configured_auth: FastAPI, client_factory: Callable[[FastAPI], TestClient]
) -> TestClient:
    """Test client with configured auth token."""
    return client_factory(app_configured_auth)


@pytest.fixture
def client_credentials_auth(
    app_credentials_auth: FastAPI, client_factory: Callable[[FastAPI], TestClient]
) -> TestClient:
    """Test client with credentials-based authentication."""
    return client_factory(app_credentials_auth)


# Auth Utilities
@pytest.fixture
def auth_test_utils() -> dict[str, Any]:
    """Utilities for auth testing.

    Returns a collection of helper functions for auth testing.
    """

    def is_auth_error(response: httpx.Response) -> bool:
        """Check if response is an authentication error."""
        return response.status_code == 401

    def is_auth_success(response: httpx.Response) -> bool:
        """Check if response indicates successful authentication."""
        return response.status_code not in [401, 403]

    def extract_auth_error_detail(response: httpx.Response) -> str | None:
        """Extract authentication error detail from response."""
        if response.status_code == 401:
            try:
                detail = response.json().get("detail")
                return str(detail) if detail is not None else None
            except Exception:
                return response.text
        return None

    return {
        "is_auth_error": is_auth_error,
        "is_auth_success": is_auth_success,
        "extract_auth_error_detail": extract_auth_error_detail,
    }


# OAuth Mock Utilities
@pytest.fixture
def oauth_flow_simulator() -> dict[str, Any]:
    """Utilities for simulating OAuth flows in tests.

    Returns functions for simulating different OAuth scenarios.
    """

    def simulate_successful_oauth() -> dict[str, str]:
        """Simulate a successful OAuth flow."""
        return {
            "access_token": "oauth-access-token-12345",
            "refresh_token": "oauth-refresh-token-67890",
            "token_type": "Bearer",
            "expires_in": "3600",
        }

    def simulate_oauth_error() -> dict[str, str]:
        """Simulate an OAuth error response."""
        return {
            "error": "invalid_grant",
            "error_description": "The provided authorization grant is invalid",
        }

    def simulate_token_refresh() -> dict[str, str]:
        """Simulate a successful token refresh."""
        return {
            "access_token": "refreshed-access-token-99999",
            "refresh_token": "new-refresh-token-11111",
            "token_type": "Bearer",
            "expires_in": "3600",
        }

    return {
        "successful_oauth": simulate_successful_oauth,
        "oauth_error": simulate_oauth_error,
        "token_refresh": simulate_token_refresh,
    }


# Docker test fixtures


@pytest.fixture
def mock_docker_run_success() -> Generator[Any, None, None]:
    """Mock asyncio.create_subprocess_exec for Docker availability check (success)."""
    from unittest.mock import AsyncMock, patch

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"Docker version 20.0.0", b"")
    mock_process.wait.return_value = 0

    with patch(
        "asyncio.create_subprocess_exec", return_value=mock_process
    ) as mock_subprocess:
        yield mock_subprocess


@pytest.fixture
def mock_docker_run_unavailable() -> Generator[Any, None, None]:
    """Mock asyncio.create_subprocess_exec for Docker availability check (unavailable)."""
    from unittest.mock import patch

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("docker: command not found"),
    ) as mock_subprocess:
        yield mock_subprocess


@pytest.fixture
def mock_docker_popen_success() -> Generator[Any, None, None]:
    """Mock asyncio.create_subprocess_exec for Docker command execution (success)."""
    from unittest.mock import AsyncMock, patch

    # Mock async stream reader
    mock_stdout = AsyncMock()
    mock_stdout.readline = AsyncMock(side_effect=[b"mock docker output\n", b""])

    mock_stderr = AsyncMock()
    mock_stderr.readline = AsyncMock(side_effect=[b""])

    mock_proc = AsyncMock()
    mock_proc.returncode = 0
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = mock_stderr
    # Also support communicate() for availability checks
    mock_proc.communicate = AsyncMock(return_value=(b"Docker version 20.0.0", b""))

    with patch(
        "asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_subprocess:
        yield mock_subprocess


@pytest.fixture
def mock_docker_popen_failure() -> Generator[Any, None, None]:
    """Mock asyncio.create_subprocess_exec for Docker command execution (failure)."""
    from unittest.mock import AsyncMock, patch

    # Mock async stream reader
    mock_stdout = AsyncMock()
    mock_stdout.readline = AsyncMock(side_effect=[b""])

    mock_stderr = AsyncMock()
    mock_stderr.readline = AsyncMock(
        side_effect=[b"docker: error running command\n", b""]
    )

    mock_proc = AsyncMock()
    mock_proc.returncode = 1
    mock_proc.wait = AsyncMock(return_value=1)
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = mock_stderr
    # Also support communicate() for availability checks
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"docker: error running command\n")
    )

    with patch(
        "asyncio.create_subprocess_exec", return_value=mock_proc
    ) as mock_subprocess:
        yield mock_subprocess


@pytest.fixture
def docker_adapter_success(
    mock_docker_run_success: Any, mock_docker_popen_success: Any
) -> DockerAdapter:
    """Create a DockerAdapter with successful subprocess mocking.

    Returns a DockerAdapter instance that will succeed on Docker operations.
    """
    from ccproxy.docker.adapter import DockerAdapter

    return DockerAdapter()


@pytest.fixture
def docker_adapter_unavailable(mock_docker_run_unavailable: Any) -> DockerAdapter:
    """Create a DockerAdapter with Docker unavailable mocking.

    Returns a DockerAdapter instance that simulates Docker not being available.
    """
    from ccproxy.docker.adapter import DockerAdapter

    return DockerAdapter()


@pytest.fixture
def docker_adapter_failure(
    mock_docker_run_success: Any, mock_docker_popen_failure: Any
) -> DockerAdapter:
    """Create a DockerAdapter with Docker failure mocking.

    Returns a DockerAdapter instance that simulates Docker command failures.
    """
    from ccproxy.docker.adapter import DockerAdapter

    return DockerAdapter()


@pytest.fixture
def docker_path_fixture(tmp_path: Path) -> DockerPath:
    """Create a DockerPath instance with temporary paths for testing.

    Returns a DockerPath configured with test directories.
    """
    from ccproxy.docker.docker_path import DockerPath

    host_path = tmp_path / "host_dir"
    host_path.mkdir()

    return DockerPath(
        host_path=host_path,
        container_path="/app/data",
        env_definition_variable_name="DATA_PATH",
    )


@pytest.fixture
def docker_path_set_fixture(tmp_path: Path) -> DockerPathSet:
    """Create a DockerPathSet instance with temporary paths for testing.

    Returns a DockerPathSet configured with test directories.
    """
    from ccproxy.docker.docker_path import DockerPathSet

    # Create multiple test directories
    host_dir1 = tmp_path / "host_dir1"
    host_dir2 = tmp_path / "host_dir2"
    host_dir1.mkdir()
    host_dir2.mkdir()

    # Create a DockerPathSet and add paths to it
    path_set = DockerPathSet(tmp_path)
    path_set.add("data1", "/app/data1", "host_dir1")
    path_set.add("data2", "/app/data2", "host_dir2")

    return path_set


@pytest.fixture
def docker_user_context() -> DockerUserContext:
    """Create a DockerUserContext for testing.

    Returns a DockerUserContext with test configuration.
    """
    from ccproxy.docker.models import DockerUserContext

    return DockerUserContext.create_manual(
        uid=1000,
        gid=1000,
        username="testuser",
        enable_user_mapping=True,
    )


@pytest.fixture
def output_middleware() -> DefaultOutputMiddleware:
    """Create a basic OutputMiddleware for testing.

    Returns a DefaultOutputMiddleware instance.
    """
    from ccproxy.docker.stream_process import DefaultOutputMiddleware

    return DefaultOutputMiddleware()


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


@pytest.fixture
def app_factory_basic(test_settings: Settings) -> FastAPI:
    """Legacy compatibility fixture - basic app via factory."""
    from tests.factories import FastAPIAppFactory

    factory = FastAPIAppFactory(default_settings=test_settings)
    return factory.create_app()


@pytest.fixture
def client_factory_basic(app_factory_basic: FastAPI) -> TestClient:
    """Legacy compatibility fixture - basic client via factory."""
    return TestClient(app_factory_basic)


# Missing fixtures for API tests compatibility


@pytest.fixture
def client_with_mock_claude(
    test_settings: Settings,
    mock_claude_service: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with mocked Claude service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_claude_service,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def client_with_mock_claude_streaming(
    test_settings: Settings,
    mock_claude_service_streaming: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with mocked Claude streaming service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_claude_service_streaming,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def client_with_unavailable_claude(
    test_settings: Settings,
    mock_claude_service_unavailable: AsyncMock,
    fastapi_app_factory: "FastAPIAppFactory",
) -> TestClient:
    """Test client with unavailable Claude service (no auth)."""
    app = fastapi_app_factory.create_app(
        settings=test_settings,
        claude_service_mock=mock_claude_service_unavailable,
        auth_enabled=False,
    )
    return TestClient(app)


@pytest.fixture
def client_with_auth(app_bearer_auth: FastAPI) -> TestClient:
    """Test client with authentication enabled."""
    return TestClient(app_bearer_auth)


@pytest.fixture
def auth_headers(
    auth_mode_bearer_token: dict[str, Any],
    auth_headers_factory: Callable[[dict[str, Any]], dict[str, str]],
) -> dict[str, str]:
    """Auth headers for bearer token authentication."""
    return auth_headers_factory(auth_mode_bearer_token)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Basic test client."""
    return TestClient(app)


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
