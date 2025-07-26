"""Tests for confirmation REST/SSE API routes."""

import asyncio
import json
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from ccproxy.api.routes.permissions import (
    event_generator,
    router,
)
from ccproxy.api.services.permission_service import (
    PermissionService,
    get_permission_service,
)
from ccproxy.config.settings import Settings, get_settings
from ccproxy.models.permissions import (
    PermissionRequest,
    PermissionStatus,
)


@pytest.fixture
def mock_confirmation_service() -> Mock:
    """Create a mock confirmation service."""
    service = Mock(spec=PermissionService)
    service.subscribe_to_events = AsyncMock()
    service.unsubscribe_from_events = AsyncMock()
    service.get_request = AsyncMock()
    service.get_status = AsyncMock()
    service.resolve = AsyncMock()
    service.request_permission = AsyncMock()
    service.wait_for_confirmation = AsyncMock()
    return service


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.server = Mock()
    settings.server.host = "localhost"
    settings.server.port = 8080
    settings.security = Mock()
    settings.security.auth_token = None  # No auth by default
    return settings


@pytest.fixture
def app(mock_settings: Settings) -> FastAPI:
    """Create a test FastAPI app."""
    app = FastAPI()

    # Override settings dependency
    app.dependency_overrides[get_settings] = lambda: mock_settings

    # Include router
    app.include_router(router)

    return app


@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def patch_confirmation_service(test_func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to patch get_permission_service for tests."""

    def wrapper(
        self: Any, test_client: TestClient, mock_confirmation_service: Any
    ) -> Any:
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service
            return test_func(self, test_client, mock_confirmation_service)

    return wrapper


class TestConfirmationRoutes:
    """Test cases for confirmation API routes."""

    @patch_confirmation_service
    def test_get_confirmation_found(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test getting an existing confirmation request."""
        # Setup mock
        from datetime import datetime, timedelta

        now = datetime.utcnow()
        mock_request = PermissionRequest(
            tool_name="bash",
            input={"command": "ls -la"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        # Create an async function that returns the request
        async def mock_get_request(confirmation_id: str):
            return mock_request

        mock_confirmation_service.get_request.side_effect = mock_get_request

        # Make request
        response = test_client.get("/test-id")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == mock_request.id
        assert data["tool_name"] == "bash"
        assert data["input"] == {"command": "ls -la"}
        assert data["status"] == "pending"

    @patch_confirmation_service
    def test_get_confirmation_not_found(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test getting a non-existent confirmation request."""
        # Setup mock
        mock_confirmation_service.get_request.return_value = None

        # Make request
        response = test_client.get("/non-existent-id")

        # Verify
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch_confirmation_service
    def test_respond_to_confirmation_allowed(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test responding to allow a confirmation request."""
        # Setup mock
        mock_confirmation_service.get_status.return_value = PermissionStatus.PENDING
        mock_confirmation_service.resolve.return_value = True

        # Make request
        response = test_client.post(
            "/test-id/respond",
            json={"allowed": True},
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["permission_id"] == "test-id"
        assert data["allowed"] is True

        # Verify service was called
        mock_confirmation_service.resolve.assert_called_once_with("test-id", True)

    @patch_confirmation_service
    def test_respond_to_confirmation_denied(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test responding to deny a confirmation request."""
        # Setup mock
        mock_confirmation_service.get_status.return_value = PermissionStatus.PENDING
        mock_confirmation_service.resolve.return_value = True

        # Make request
        response = test_client.post(
            "/test-id/respond",
            json={"allowed": False},
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["permission_id"] == "test-id"
        assert data["allowed"] is False

        # Verify service was called
        mock_confirmation_service.resolve.assert_called_once_with("test-id", False)

    @patch_confirmation_service
    def test_respond_to_non_existent_confirmation(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test responding to a non-existent confirmation request."""
        # Setup mock
        mock_confirmation_service.get_status.return_value = None

        # Make request
        response = test_client.post(
            "/non-existent-id/respond",
            json={"allowed": True},
        )

        # Verify
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch_confirmation_service
    def test_respond_to_already_resolved_confirmation(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test responding to an already resolved confirmation."""
        # Setup mock
        mock_confirmation_service.get_status.return_value = PermissionStatus.ALLOWED

        # Make request
        response = test_client.post(
            "/test-id/respond",
            json={"allowed": False},
        )

        # Verify
        assert response.status_code == 409
        assert "already resolved" in response.json()["detail"].lower()

    @patch_confirmation_service
    def test_respond_resolution_failure(
        self,
        test_client: TestClient,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test when resolve returns False (shouldn't happen but handled)."""
        # Setup mock
        mock_confirmation_service.get_status.return_value = PermissionStatus.PENDING
        mock_confirmation_service.resolve.return_value = False

        # Make request
        response = test_client.post(
            "/test-id/respond",
            json={"allowed": True},
        )

        # Verify
        assert response.status_code == 409
        assert "Failed to resolve" in response.json()["detail"]


class TestSSEEventGenerator:
    """Test cases for SSE event generation."""

    @pytest.fixture
    def mock_request(self) -> Mock:
        """Create a mock request object."""
        request = Mock(spec=Request)
        request.is_disconnected = AsyncMock(return_value=False)
        return request

    async def test_event_generator_initial_ping(
        self,
        mock_request: Mock,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test that event generator sends initial ping."""
        # Setup mock queue
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        mock_confirmation_service.subscribe_to_events.return_value = queue

        # Patch get_permission_service at module level
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Get first event
            generator = event_generator(mock_request)
            first_event = await generator.__anext__()

            # Verify initial ping
            assert first_event["event"] == "ping"
            data = json.loads(first_event["data"])
            assert "Connected" in data["message"]

            # Cleanup
            await generator.aclose()

    @pytest.mark.skip("timeout")
    async def test_event_generator_forwards_events(
        self,
        mock_request: Mock,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test that event generator forwards events from queue."""
        # Setup mock queue with event
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        test_event = {
            "type": "confirmation_request",
            "request_id": "test-id",
            "tool_name": "bash",
            "input": {"command": "ls"},
        }
        await queue.put(test_event)

        mock_confirmation_service.subscribe_to_events.return_value = queue

        # Setup request to disconnect after getting event
        call_count = 0

        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 2  # Disconnect after initial ping and first event

        mock_request.is_disconnected = is_disconnected

        # Patch get_permission_service
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Get events
            generator = event_generator(mock_request)
            events = []
            async for event in generator:
                events.append(event)

            # Verify - May get keepalive pings
            assert len(events) >= 2  # At least initial ping + test event

            # Find the confirmation request event
            confirmation_event = None
            for event in events:
                if event["event"] == "confirmation_request":
                    confirmation_event = event
                    break

            assert confirmation_event is not None
            data = json.loads(confirmation_event["data"])
            assert data["request_id"] == "test-id"
            assert data["tool_name"] == "bash"

    async def test_event_generator_keepalive(
        self,
        mock_request: Mock,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test that event generator sends keepalive pings."""
        # Setup empty queue
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        mock_confirmation_service.subscribe_to_events.return_value = queue

        # Setup request to disconnect after keepalive
        call_count = 0

        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 2

        mock_request.is_disconnected = is_disconnected

        # Patch get_permission_service
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Get events with short timeout
            generator = event_generator(mock_request)
            events = []

            # Patch wait_for to simulate timeout quickly
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                async for event in generator:
                    events.append(event)
                    if len(events) >= 2:  # Initial ping + keepalive
                        break

            # Verify keepalive
            assert len(events) >= 2
            assert events[0]["event"] == "ping"  # Initial
            assert events[1]["event"] == "ping"  # Keepalive
            data = json.loads(events[1]["data"])
            assert data["message"] == "keepalive"

    @pytest.mark.skip("timeout")
    async def test_event_generator_cleanup_on_disconnect(
        self,
        mock_request: Mock,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test that event generator cleans up when client disconnects."""
        # Setup mock queue
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        mock_confirmation_service.subscribe_to_events.return_value = queue

        # Setup request to disconnect immediately after ping
        call_count = 0

        async def is_disconnected():
            nonlocal call_count
            call_count += 1
            return call_count > 1  # Disconnect after initial ping

        mock_request.is_disconnected = is_disconnected

        # Patch get_permission_service
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Run generator
            generator = event_generator(mock_request)
            events = []
            async for event in generator:
                events.append(event)

            # Verify cleanup was called
            mock_confirmation_service.unsubscribe_from_events.assert_called_once_with(
                queue
            )

    async def test_event_generator_handles_cancellation(
        self,
        mock_request: Mock,
        mock_confirmation_service: Mock,
    ) -> None:
        """Test that event generator handles cancellation gracefully."""
        # Setup mock queue
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        mock_confirmation_service.subscribe_to_events.return_value = queue

        # Patch get_permission_service
        with patch(
            "ccproxy.api.routes.permissions.get_permission_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Create generator
            generator = event_generator(mock_request)

            # Get initial ping
            await generator.__anext__()

            # Cancel generator
            await generator.aclose()

            # Verify cleanup
            mock_confirmation_service.unsubscribe_from_events.assert_called_once_with(
                queue
            )


@pytest.mark.skip(
    reason="SSE endpoint creates endless stream, tested via event_generator"
)
@pytest.mark.asyncio
async def test_sse_stream_endpoint(
    mock_confirmation_service: Mock, mock_settings: Settings
) -> None:
    """Test the SSE stream endpoint with async client."""
    from fastapi import FastAPI

    from ccproxy.config.settings import get_settings

    app = FastAPI()
    app.include_router(router)

    # Override dependencies
    app.dependency_overrides[get_permission_service] = lambda: mock_confirmation_service
    app.dependency_overrides[get_settings] = lambda: mock_settings

    # Setup mock queue
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    mock_confirmation_service.subscribe_to_events.return_value = queue

    # Use TestClient for SSE since httpx AsyncClient needs a running server
    with TestClient(app) as test_client:
        # Just verify the endpoint responds correctly
        # Streaming behavior is tested in event_generator tests
        response = test_client.get(
            "/stream",
            headers={"Accept": "text/event-stream"},
        )
        assert response.status_code == 200
        # Headers are set by EventSourceResponse which TestClient doesn't fully support
