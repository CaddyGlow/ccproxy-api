"""Shared test fixtures and configuration for ccproxy tests.

This module provides minimal, focused fixtures for testing the ccproxy application.
All fixtures have proper type hints and are designed to work with real components
while mocking only external services.
"""

import json
from collections.abc import AsyncGenerator, Generator

# Override settings for testing
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pytest_httpx import HTTPXMock

from ccproxy.api.app import create_app
from ccproxy.config.auth import AuthSettings, CredentialStorageSettings
from ccproxy.config.security import SecuritySettings
from ccproxy.config.server import ServerSettings
from ccproxy.config.settings import Settings, get_settings
from ccproxy.docker.adapter import DockerAdapter
from ccproxy.docker.docker_path import DockerPath, DockerPathSet
from ccproxy.docker.models import DockerUserContext
from ccproxy.docker.stream_process import DefaultOutputMiddleware
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage


@lru_cache
def get_test_settings(test_settings: Settings) -> Settings:
    """Get test settings - overrides the default settings provider."""
    return test_settings


# Test data directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create isolated test settings with temp directories.

    Returns a Settings instance configured for testing with:
    - Temporary config and cache directories
    - In-memory metrics storage
    - No authentication by default
    - Test environment enabled
    """
    return Settings(
        server=ServerSettings(log_level="WARNING"),
        security=SecuritySettings(auth_token=None),  # No auth by default
        auth=AuthSettings(
            storage=CredentialStorageSettings(storage_paths=[tmp_path / ".claude/"])
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
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    return app


@pytest.fixture
def mock_claude_service() -> AsyncMock:
    """Create a mock Claude SDK service for testing."""
    mock_service = AsyncMock()

    # Mock the create_completion method for non-streaming
    mock_service.create_completion.return_value = {
        "id": "msg_01234567890",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help you?"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 8},
    }

    # Mock the list_models method
    mock_service.list_models.return_value = [
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
    ]

    # Mock the validate_health method
    mock_service.validate_health.return_value = True

    return mock_service


@pytest.fixture
def mock_claude_service_unavailable() -> AsyncMock:
    """Create a mock Claude SDK service that simulates CLI unavailability."""
    from ccproxy.core.errors import ServiceUnavailableError

    mock_service = AsyncMock()

    # Mock methods to raise ServiceUnavailableError
    mock_service.create_completion.side_effect = ServiceUnavailableError(
        "Claude CLI not available"
    )
    mock_service.list_models.side_effect = ServiceUnavailableError(
        "Claude CLI not available"
    )
    mock_service.validate_health.side_effect = ServiceUnavailableError(
        "Claude CLI not available"
    )

    return mock_service


@pytest.fixture
def app_with_unavailable_claude(
    test_settings: Settings, mock_claude_service_unavailable: AsyncMock
) -> FastAPI:
    """Create test FastAPI application with unavailable Claude service.

    Returns a configured FastAPI app that simulates Claude CLI unavailability.
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.api.dependencies import get_claude_service
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    # Override dependency with a function that returns unavailable service
    def mock_get_claude_service_unavailable(
        auth_manager: Any = None, metrics_collector: Any = None
    ) -> AsyncMock:
        return mock_claude_service_unavailable

    app.dependency_overrides[get_claude_service] = mock_get_claude_service_unavailable

    return app


@pytest.fixture
def client_with_unavailable_claude(app_with_unavailable_claude: FastAPI) -> TestClient:
    """Create client for testing with unavailable Claude service.

    Returns a TestClient that simulates Claude CLI unavailability.
    """
    return TestClient(app_with_unavailable_claude)


@pytest.fixture
def mock_claude_service_streaming() -> AsyncMock:
    """Create a mock Claude SDK service for streaming tests."""

    async def mock_streaming_response() -> AsyncGenerator[dict[str, Any], None]:
        """Mock streaming response generator."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

        for event in events:
            yield event  # type: ignore[misc]

    mock_service = AsyncMock()

    # Mock create_completion as an async function
    async def mock_create_completion(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream", False):
            return mock_streaming_response()
        else:
            return {
                "id": "msg_01234567890",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help you?"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }

    mock_service.create_completion = mock_create_completion

    # Mock the list_models method
    mock_service.list_models.return_value = [
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
    ]

    # Mock the validate_health method
    mock_service.validate_health.return_value = True

    return mock_service


@pytest.fixture
def app_with_mock_claude_streaming(
    test_settings: Settings, mock_claude_service_streaming: AsyncMock
) -> FastAPI:
    """Create test FastAPI application with mocked Claude streaming service.

    Returns a configured FastAPI app with mocked streaming dependencies.
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.api.dependencies import get_claude_service
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    # Override dependency with a function that accepts dependencies but returns mock
    def mock_get_claude_service_streaming(
        auth_manager: Any = None, metrics_collector: Any = None
    ) -> AsyncMock:
        return mock_claude_service_streaming

    app.dependency_overrides[get_claude_service] = mock_get_claude_service_streaming

    return app


@pytest.fixture
def client_with_mock_claude_streaming(
    app_with_mock_claude_streaming: FastAPI,
) -> TestClient:
    """Create test client with mocked Claude streaming service.

    Returns a TestClient for making requests with mocked streaming.
    """
    return TestClient(app_with_mock_claude_streaming)


@pytest_asyncio.fixture
async def async_client_with_mock_claude_streaming(
    app_with_mock_claude_streaming: FastAPI,
) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with mocked Claude streaming service.

    Yields an AsyncClient for making async requests with mocked streaming.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app_with_mock_claude_streaming),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def app_with_mock_claude(
    test_settings: Settings, mock_claude_service: AsyncMock
) -> FastAPI:
    """Create test FastAPI application with mocked Claude service.

    Returns a configured FastAPI app with mocked dependencies.
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.api.dependencies import get_claude_service
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    # Override dependency with a function that accepts dependencies but returns mock
    def mock_get_claude_service(
        auth_manager: Any = None, metrics_collector: Any = None
    ) -> AsyncMock:
        return mock_claude_service

    app.dependency_overrides[get_claude_service] = mock_get_claude_service

    return app


@pytest.fixture
def client_with_mock_claude(app_with_mock_claude: FastAPI) -> TestClient:
    """Create client for testing with mocked Claude service.

    Returns a TestClient with mocked Claude service dependencies.
    """
    return TestClient(app_with_mock_claude)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create synchronous test client for API testing.

    Returns a TestClient for making synchronous HTTP requests.
    """
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client for async API testing.

    Yields an AsyncClient for making asynchronous HTTP requests.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def auth_settings(test_settings: Settings) -> Settings:
    """Create settings with authentication enabled.

    Returns Settings configured with a test auth token.
    """
    test_settings.security.auth_token = "test-token-12345"
    return test_settings


@pytest.fixture
def app_with_auth(auth_settings: Settings) -> FastAPI:
    """Create app with authentication enabled.

    Returns a FastAPI app that requires authentication.
    """
    # Create app
    app = create_app(settings=auth_settings)

    # Override the settings dependency for testing
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: auth_settings

    return app


@pytest.fixture
def client_with_auth(app_with_auth: FastAPI) -> TestClient:
    """Create client for testing authenticated endpoints.

    Returns a TestClient with auth-enabled app.
    """
    return TestClient(app_with_auth)


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
def mock_claude(httpx_mock: HTTPXMock, claude_responses: dict[str, Any]) -> HTTPXMock:
    """Mock Claude API responses for standard completion.

    Returns HTTPXMock configured with Claude API responses.
    """
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json=claude_responses["standard_completion"],
        status_code=200,
        headers={"content-type": "application/json"},
    )
    return httpx_mock


@pytest.fixture
def mock_claude_stream(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock Claude API streaming responses.

    Returns HTTPXMock configured for SSE streaming.
    """

    def stream_generator() -> Generator[str, None, None]:
        """Generate SSE formatted streaming response."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

        for event in events:
            yield f"data: {json.dumps(event)}\n\n"

    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        content=b"".join(chunk.encode() for chunk in stream_generator()),
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
        },
    )
    return httpx_mock


@pytest.fixture
def mock_oauth(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock OAuth token endpoints.

    Returns HTTPXMock configured with OAuth responses.
    """
    # Mock token exchange
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/token",
        json={
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    # Mock token refresh
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/refresh",
        json={
            "access_token": "new_test_access_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    return httpx_mock


@pytest.fixture
def metrics_storage() -> InMemoryMetricsStorage:
    """Create isolated in-memory metrics storage.

    Returns an InMemoryMetricsStorage instance for testing.
    """
    return InMemoryMetricsStorage()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Standard authentication headers for testing.

    Returns headers with test auth token.
    """
    return {"Authorization": "Bearer test-token-12345"}


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
    from ccproxy.docker.docker_path import DockerPath, DockerPathSet

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
