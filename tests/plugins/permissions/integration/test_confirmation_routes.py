"""Integration tests for the confirmation system."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from ccproxy.api.app import create_app, initialize_plugins_startup
from ccproxy.api.bootstrap import create_service_container
from ccproxy.api.dependencies import get_cached_settings
from ccproxy.auth.dependencies import get_conditional_auth_manager
from ccproxy.config.core import LoggingSettings
from ccproxy.config.settings import Settings
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.plugins.permissions.models import PermissionStatus
from ccproxy.plugins.permissions.routes import router as confirmation_router
from ccproxy.plugins.permissions.service import (
    PermissionService,
    get_permission_service,
)
from ccproxy.services.container import ServiceContainer


@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Start and stop task manager for each test."""
    container = ServiceContainer.get_current(strict=False)
    if container is None:
        container = create_service_container()
    await start_task_manager(container=container)
    try:
        yield
    finally:
        await stop_task_manager(container=container)


@pytest.fixture
async def confirmation_service() -> AsyncGenerator[PermissionService, None]:
    """Create and start a test confirmation service."""
    service = PermissionService(timeout_seconds=5)
    await service.start()
    yield service
    await service.stop()


@pytest.fixture
async def app(confirmation_service: PermissionService) -> FastAPI:
    """Create a FastAPI app with real confirmation service."""
    from pydantic import BaseModel

    settings = Settings()
    container = ServiceContainer(settings)
    container.register_service(PermissionService, instance=confirmation_service)

    enabled_plugins = ["permissions"]
    plugin_configs = {"permissions": {"enabled": True}}
    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=True,
        enabled_plugins=enabled_plugins,
        plugins=plugin_configs,
        logging=LoggingSettings(
            **{
                "level": "TRACE",
                "verbose_api": False,
            }
        ),
    )

    # setup_logging(json_logs=False, log_level_name="DEBUG")

    service_container = create_service_container(settings)
    app = create_app(service_container)
    await initialize_plugins_startup(app, settings)

    # app = FastAPI()
    # app.state.service_container = container
    app.include_router(confirmation_router, prefix="/confirmations")

    class MCPRequest(BaseModel):
        tool: str
        input: dict[str, str]

    @app.post("/api/v1/mcp/check-permission")
    async def check_permission(request: MCPRequest) -> dict[str, Any]:
        """Test MCP endpoint that mimics the real one."""
        if not request.tool:
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="Tool name is required")

        service = container.get_service(PermissionService)
        confirmation_id = await service.request_permission(
            tool_name=request.tool,
            input=request.input,
        )

        return {
            "confirmationId": confirmation_id,
            "message": "Confirmation required. Please check the terminal or confirmation UI.",
        }

    return app


@pytest_asyncio.fixture
async def test_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Create an async HTTP client bound to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


class TestConfirmationIntegration:
    """Integration tests for the confirmation system."""

    @patch("ccproxy.plugins.permissions.routes.get_permission_service")
    async def test_mcp_permission_flow(
        self,
        mock_get_service: Mock,
        test_client: AsyncClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test the full MCP permission flow."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Subscribe to events
        event_queue = await confirmation_service.subscribe_to_events()

        # Make MCP permission request
        mcp_response = await test_client.post(
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
        get_response = await test_client.get(f"/confirmations/{confirmation_id}")
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["status"] == "pending"
        assert get_data["tool_name"] == "bash"

        # Approve confirmation
        approve_response = await test_client.post(
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
        confirmation_service: PermissionService,
    ) -> None:
        """Test SSE streaming with multiple clients."""
        # For SSE streaming tests, we'll use the confirmation service directly
        # since the async client won't consume SSE streams automatically

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

    @patch("ccproxy.plugins.permissions.routes.get_permission_service")
    async def test_confirmation_expiration(
        self,
        mock_get_service: Mock,
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

            test_settings = Settings()

            def _override_settings(_: Request) -> Settings:
                return test_settings

            async def _override_auth_manager(_: Request) -> None:
                return None

            app.dependency_overrides[get_cached_settings] = _override_settings
            app.dependency_overrides[get_conditional_auth_manager] = (
                _override_auth_manager
            )

            # Create confirmation
            request_id = await service.request_permission("bash", {"command": "test"})

            # Force the request to expire without waiting for the poller.
            permission = await service.get_request(request_id)
            assert permission is not None
            permission.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            permission._resolved_event.set()

            # Try to respond - should fail
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                response = await client.post(
                    f"/confirmations/{request_id}/respond",
                    json={"allowed": True},
                )

            # Should get conflict since it's expired
            assert response.status_code == 409

        finally:
            await service.stop()

    @patch("ccproxy.plugins.permissions.routes.get_permission_service")
    async def test_concurrent_confirmations(
        self,
        mock_get_service: Mock,
        test_client: AsyncClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test handling multiple concurrent confirmations."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Create multiple confirmation requests
        request_ids = []
        for i in range(5):
            response = await test_client.post(
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
            response = await test_client.post(
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

    @patch("ccproxy.plugins.permissions.routes.get_permission_service")
    async def test_duplicate_resolution_attempts(
        self,
        mock_get_service: Mock,
        test_client: AsyncClient,
        confirmation_service: PermissionService,
    ) -> None:
        """Test that duplicate resolution attempts are rejected."""
        # Make the patched function return our test service
        mock_get_service.return_value = confirmation_service

        # Create confirmation
        response = await test_client.post(
            "/api/v1/mcp/check-permission",
            json={
                "tool": "bash",
                "input": {"command": "test"},
            },
        )
        request_id = response.json()["confirmationId"]

        # First resolution should succeed
        response1 = await test_client.post(
            f"/confirmations/{request_id}/respond",
            json={"allowed": True},
        )
        assert response1.status_code == 200

        # Second resolution should fail
        response2 = await test_client.post(
            f"/confirmations/{request_id}/respond",
            json={"allowed": False},
        )
        assert response2.status_code == 409

        error_body = response2.json()
        detail = error_body.get("detail")
        if detail is None:
            detail = error_body.get("error", {}).get("message")

        assert detail is not None
        assert "already resolved" in detail.lower()

        # Status should still be allowed (from first resolution)
        status = await confirmation_service.get_status(request_id)
        assert status == PermissionStatus.ALLOWED


class TestConfirmationEdgeCases:
    """Test edge cases and error conditions."""

    async def test_invalid_mcp_request(self, test_client: AsyncClient) -> None:
        """Test MCP endpoint with invalid input."""
        # Missing tool name
        response = await test_client.post(
            "/api/v1/mcp/check-permission",
            json={"input": {"command": "test"}},
        )
        assert response.status_code == 422

        # Empty tool name
        response = await test_client.post(
            "/api/v1/mcp/check-permission",
            json={"tool": "", "input": {"command": "test"}},
        )
        assert response.status_code == 400

    async def test_confirmation_api_validation(self, test_client: AsyncClient) -> None:
        """Test confirmation API input validation."""
        # Invalid confirmation ID format
        response = await test_client.get("/confirmations/")
        assert response.status_code == 404

        # Missing allowed field
        response = await test_client.post(
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
