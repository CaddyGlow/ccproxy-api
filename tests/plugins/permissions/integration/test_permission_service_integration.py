"""Integration tests for permission service functionality."""

import asyncio

import pytest

from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.core.errors import PermissionNotFoundError
from ccproxy.plugins.permissions.models import PermissionStatus
from ccproxy.plugins.permissions.service import (
    PermissionService,
    get_permission_service,
)


@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Start and stop task manager for each test."""
    await start_task_manager()
    yield
    await stop_task_manager()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_request_creates_request(
    disabled_plugins_client,
) -> None:
    """Test that requesting permission creates a new request."""
    # Create a fresh service for this test
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        tool_name = "bash"
        input_params = {"command": "ls -la"}

        request_id = await service.request_permission(tool_name, input_params)

        assert request_id is not None
        assert len(request_id) > 0

        # Check request was stored
        request = await service.get_request(request_id)
        assert request is not None
        assert request.tool_name == tool_name
        assert request.input == input_params
        assert request.status == PermissionStatus.PENDING
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_validates_input(
    disabled_plugins_client,
) -> None:
    """Test input validation for permission requests."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        # Test empty tool name
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            await service.request_permission("", {"command": "test"})

        # Test whitespace-only tool name
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            await service.request_permission("   ", {"command": "test"})

        # Test None input
        with pytest.raises(ValueError, match="Input parameters cannot be None"):
            await service.request_permission("bash", None)  # type: ignore
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_get_status(
    disabled_plugins_client,
) -> None:
    """Test getting status of permission requests."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # Check initial status
        status = await service.get_status(request_id)
        assert status == PermissionStatus.PENDING

        # Check non-existent request
        status = await service.get_status("non-existent-id")
        assert status is None
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_resolve_allowed(
    disabled_plugins_client,
) -> None:
    """Test resolving a permission request as allowed."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # Resolve as allowed
        success = await service.resolve(request_id, allowed=True)
        assert success is True

        # Check status updated
        status = await service.get_status(request_id)
        assert status == PermissionStatus.ALLOWED
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_resolve_denied(
    disabled_plugins_client,
) -> None:
    """Test resolving a permission request as denied."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # Resolve as denied
        success = await service.resolve(request_id, allowed=False)
        assert success is True

        # Check status updated
        status = await service.get_status(request_id)
        assert status == PermissionStatus.DENIED
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_resolve_validation(
    disabled_plugins_client,
) -> None:
    """Test input validation for resolve method."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        # Test empty request ID
        with pytest.raises(ValueError, match="Request ID cannot be empty"):
            await service.resolve("", True)

        # Non-existent request should return False (not raise exception)
        success = await service.resolve("non-existent-id", True)
        assert success is False
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_resolve_already_resolved(
    disabled_plugins_client,
) -> None:
    """Test resolving an already resolved request returns False."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # First resolution succeeds
        success = await service.resolve(request_id, True)
        assert success is True

        # Second resolution fails
        success = await service.resolve(request_id, False)
        assert success is False
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_concurrent_resolutions(
    disabled_plugins_client,
) -> None:
    """Test handling concurrent resolution attempts."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # Attempt concurrent resolutions
        results = await asyncio.gather(
            service.resolve(request_id, True),
            service.resolve(request_id, False),
            return_exceptions=True,
        )

        # Only one should succeed
        successes = [r for r in results if r is True]
        assert len(successes) == 1
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_event_subscription(
    disabled_plugins_client,
) -> None:
    """Test event subscription and emission."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        # Subscribe to events
        queue = await service.subscribe_to_events()

        # Create a permission request
        request_id = await service.request_permission("bash", {"command": "test"})

        # Check we received the event
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "permission_request"
        assert event["request_id"] == request_id
        assert event["tool_name"] == "bash"

        # Resolve the request
        await service.resolve(request_id, True)

        # Check we received the resolution event
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "permission_resolved"
        assert event["request_id"] == request_id
        assert event["allowed"] is True

        # Unsubscribe
        await service.unsubscribe_from_events(queue)
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_multiple_subscribers(
    disabled_plugins_client,
) -> None:
    """Test multiple event subscribers receive events."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        # Subscribe multiple queues
        queue1 = await service.subscribe_to_events()
        queue2 = await service.subscribe_to_events()

        # Create a request
        request_id = await service.request_permission("bash", {"command": "test"})

        # Both queues should receive the event
        event1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
        event2 = await asyncio.wait_for(queue2.get(), timeout=1.0)

        assert event1["request_id"] == request_id
        assert event2["request_id"] == request_id

        # Cleanup
        await service.unsubscribe_from_events(queue1)
        await service.unsubscribe_from_events(queue2)
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_wait_for_permission_timeout(
    disabled_plugins_client,
) -> None:
    """Test waiting for a permission that times out."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        request_id = await service.request_permission("bash", {"command": "test"})

        # Don't resolve - let it timeout
        with pytest.raises(asyncio.TimeoutError):
            await service.wait_for_permission(request_id, timeout_seconds=0.2)
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_wait_for_non_existent_request(
    disabled_plugins_client,
) -> None:
    """Test waiting for a non-existent request."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        with pytest.raises(PermissionNotFoundError):
            await service.wait_for_permission("non-existent-id")
    finally:
        await service.stop()


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_get_permission_service_singleton(disabled_plugins_client) -> None:
    """Test that get_permission_service returns singleton."""
    service1 = get_permission_service()
    service2 = get_permission_service()
    assert service1 is service2


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.integration
@pytest.mark.auth
async def test_permission_service_get_pending_requests(
    disabled_plugins_client,
) -> None:
    """Test get_pending_requests returns only pending requests."""
    service = PermissionService(timeout_seconds=30)
    await service.start()

    try:
        # Create multiple requests with different statuses
        request_id1 = await service.request_permission("tool1", {"param": "value1"})
        request_id2 = await service.request_permission("tool2", {"param": "value2"})
        request_id3 = await service.request_permission("tool3", {"param": "value3"})

        # Resolve one as allowed and one as denied
        await service.resolve(request_id1, True)
        await service.resolve(request_id2, False)

        # Get pending requests
        pending = await service.get_pending_requests()

        # Should only have one pending request
        assert len(pending) == 1
        assert pending[0].id == request_id3
        assert pending[0].tool_name == "tool3"
        assert pending[0].status == PermissionStatus.PENDING
    finally:
        await service.stop()
