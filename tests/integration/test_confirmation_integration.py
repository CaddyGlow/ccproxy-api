"""Integration tests for the confirmation system."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ccproxy.api.routes.permissions import router as confirmation_router
from ccproxy.api.services.permission_service import (
    PermissionService,
    get_permission_service,
)
from ccproxy.config.settings import Settings, get_settings
from ccproxy.models.permissions import PermissionStatus


@pytest.fixture
async def confirmation_service() -> AsyncGenerator[PermissionService, None]:
    """Create and start a test confirmation service."""
    service = PermissionService(timeout_seconds=5)
    await service.start()
    yield service
    await service.stop()


@pytest.fixture
def app(confirmation_service: PermissionService) -> FastAPI:
    """Create a FastAPI app with real confirmation service."""
    from pydantic import BaseModel

    app = FastAPI()
    app.include_router(confirmation_router, prefix="/confirmations")

    # Override to use test service
    app.dependency_overrides[get_permission_service] = lambda: confirmation_service

    # Mock settings
    mock_settings = Mock(spec=Settings)
    mock_settings.server = Mock()
    mock_settings.server.host = "localhost"
    mock_settings.server.port = 8080
    app.dependency_overrides[get_settings] = lambda: mock_settings

    # Add test MCP endpoint since mcp.py doesn't export a router
    class MCPRequest(BaseModel):
        tool: str
        input: dict[str, str]

    @app.post("/api/v1/mcp/check-permission")
    async def check_permission(request: MCPRequest) -> dict[str, Any]:
        """Test MCP endpoint that mimics the real one."""
        if not request.tool:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="Tool name is required")

        # Use the same confirmation service instance
        service = app.dependency_overrides[get_permission_service]()
        confirmation_id = await service.request_permission(
            tool_name=request.tool,
            input=request.input,
        )

        return {
            "confirmationId": confirmation_id,
            "message": "Confirmation required. Please check the terminal or confirmation UI.",
        }

    return app


@pytest.fixture
def test_client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestConfirmationIntegration:
    """Integration tests for the confirmation system."""

    @patch("ccproxy.api.routes.permissions.get_permission_service")
    async def test_mcp_permission_flow(
        self,
        mock_get_service: Mock,
        test_client: TestClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test the full MCP permission flow."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Subscribe to events
        event_queue = await confirmation_service.subscribe_to_events()

        # Make MCP permission request
        mcp_response = test_client.post(
            "/api/v1/mcp/check-permission",
            json={
                "tool": "bash",
                "input": {"command": "ls -la"},
            },
        )

        # Should return pending with confirmation ID
        assert mcp_response.status_code == 200
        mcp_data = mcp_response.json()
        assert "confirmationId" in mcp_data
        assert "Confirmation required" in mcp_data["message"]

        confirmation_id = mcp_data["confirmationId"]

        # Should have received event
        event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        assert event["type"] == "permission_request"
        assert event["request_id"] == confirmation_id
        assert event["tool_name"] == "bash"

        # Get confirmation details
        get_response = test_client.get(f"/confirmations/{confirmation_id}")
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["status"] == "pending"
        assert get_data["tool_name"] == "bash"

        # Approve confirmation
        approve_response = test_client.post(
            f"/confirmations/{confirmation_id}/respond",
            json={"allowed": True},
        )
        assert approve_response.status_code == 200

        # Should have received resolution event
        resolution_event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
        assert resolution_event["type"] == "permission_resolved"
        assert resolution_event["request_id"] == confirmation_id
        assert resolution_event["allowed"] is True

        # Verify status is now allowed
        status = await confirmation_service.get_status(confirmation_id)
        assert status == PermissionStatus.ALLOWED

        # Cleanup
        await confirmation_service.unsubscribe_from_events(event_queue)

    async def test_sse_streaming_multiple_clients(
        self,
        test_client: TestClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test SSE streaming with multiple clients."""
        # For SSE streaming tests, we'll use the confirmation service directly
        # since TestClient doesn't properly handle SSE streaming

        # Subscribe two event queues directly
        queue1 = await confirmation_service.subscribe_to_events()
        queue2 = await confirmation_service.subscribe_to_events()

        try:
            # Create confirmation request
            request_id = await confirmation_service.request_permission(
                "bash", {"command": "echo test"}
            )

            # Both queues should receive the event
            event1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
            event2 = await asyncio.wait_for(queue2.get(), timeout=1.0)

            # Verify both got the same event
            assert event1["type"] == "permission_request"
            assert event2["type"] == "permission_request"
            assert event1["request_id"] == request_id
            assert event2["request_id"] == request_id

        finally:
            # Cleanup
            await confirmation_service.unsubscribe_from_events(queue1)
            await confirmation_service.unsubscribe_from_events(queue2)

    @patch("ccproxy.api.routes.permissions.get_permission_service")
    async def test_confirmation_expiration(
        self,
        mock_get_service: Mock,
        test_client: TestClient,
    ) -> None:
        """Test that confirmations expire correctly."""
        # Create service with very short timeout
        service = PermissionService(timeout_seconds=1)
        await service.start()

        # Make the patched function return our test service
        mock_get_service.return_value = service

        try:
            # Override service in app
            app = FastAPI()
            app.include_router(confirmation_router, prefix="/confirmations")
            app.dependency_overrides[get_permission_service] = lambda: service

            # Create confirmation
            request_id = await service.request_permission("bash", {"command": "test"})

            # Wait for expiration
            await asyncio.sleep(2)

            # Try to respond - should fail
            with TestClient(app) as client:
                response = client.post(
                    f"/confirmations/{request_id}/respond",
                    json={"allowed": True},
                )

            # Should get conflict since it's expired
            assert response.status_code == 409

        finally:
            await service.stop()

    @patch("ccproxy.api.routes.permissions.get_permission_service")
    async def test_concurrent_confirmations(
        self,
        mock_get_service: Mock,
        test_client: TestClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test handling multiple concurrent confirmations."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Create multiple confirmation requests
        request_ids = []
        for i in range(5):
            response = test_client.post(
                "/api/v1/mcp/check-permission",
                json={
                    "tool": "bash",
                    "input": {"command": f"echo test{i}"},
                },
            )
            assert response.status_code == 200
            request_ids.append(response.json()["confirmationId"])

        # Resolve them concurrently with different responses
        async def resolve_confirmation(request_id: str, index: int) -> None:
            """Resolve a single confirmation."""
            allowed = index % 2 == 0  # Even indices allowed, odd denied
            response = test_client.post(
                f"/confirmations/{request_id}/respond",
                json={"allowed": allowed},
            )
            assert response.status_code == 200

        # Resolve all concurrently
        await asyncio.gather(
            *[resolve_confirmation(req_id, i) for i, req_id in enumerate(request_ids)]
        )

        # Verify all statuses
        for i, request_id in enumerate(request_ids):
            status = await confirmation_service.get_status(request_id)
            expected = (
                PermissionStatus.ALLOWED if i % 2 == 0 else PermissionStatus.DENIED
            )
            assert status == expected

    @patch("ccproxy.api.routes.permissions.get_permission_service")
    async def test_duplicate_resolution_attempts(
        self,
        mock_get_service: Mock,
        test_client: TestClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test that duplicate resolution attempts are rejected."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Create confirmation
        response = test_client.post(
            "/api/v1/mcp/check-permission",
            json={
                "tool": "bash",
                "input": {"command": "test"},
            },
        )
        request_id = response.json()["confirmationId"]

        # First resolution should succeed
        response1 = test_client.post(
            f"/confirmations/{request_id}/respond",
            json={"allowed": True},
        )
        assert response1.status_code == 200

        # Second resolution should fail
        response2 = test_client.post(
            f"/confirmations/{request_id}/respond",
            json={"allowed": False},
        )
        assert response2.status_code == 409
        assert "already resolved" in response2.json()["detail"].lower()

        # Status should still be allowed (from first resolution)
        status = await confirmation_service.get_status(request_id)
        assert status == PermissionStatus.ALLOWED


class TestConfirmationEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_mcp_request(self, test_client: TestClient) -> None:
        """Test MCP endpoint with invalid input."""
        # Missing tool name
        response = test_client.post(
            "/api/v1/mcp/check-permission",
            json={"input": {"command": "test"}},
        )
        assert response.status_code == 422

        # Empty tool name
        response = test_client.post(
            "/api/v1/mcp/check-permission",
            json={"tool": "", "input": {"command": "test"}},
        )
        assert response.status_code == 400

    def test_confirmation_api_validation(self, test_client: TestClient) -> None:
        """Test confirmation API input validation."""
        # Invalid confirmation ID format
        response = test_client.get("/confirmations/")
        assert response.status_code == 404

        # Missing allowed field
        response = test_client.post(
            "/confirmations/test-id/respond",
            json={},
        )
        assert response.status_code == 422

    async def test_service_shutdown_during_request(
        self,
        confirmation_service: PermissionService,
    ) -> None:
        """Test behavior when service shuts down during active requests."""
        # Create a request
        request_id = await confirmation_service.request_permission(
            "bash", {"command": "test"}
        )

        # Stop service
        await confirmation_service.stop()

        # Try to get status - should still work (data in memory)
        status = await confirmation_service.get_status(request_id)
        assert status == PermissionStatus.PENDING

        # Try to resolve - should still work
        success = await confirmation_service.resolve(request_id, True)
        assert success is True
