"""Tests for confirmation service functionality."""

import asyncio
from collections.abc import AsyncGenerator

import pytest

from ccproxy.api.services.permission_service import (
    PermissionService,
    get_permission_service,
)
from ccproxy.core.errors import (
    PermissionNotFoundError,
)
from ccproxy.models.permissions import (
    PermissionStatus,
)


@pytest.fixture
def confirmation_service() -> PermissionService:
    """Create a test confirmation service."""
    service = PermissionService(timeout_seconds=30)
    return service


@pytest.fixture
async def started_service(
    confirmation_service: PermissionService,
) -> AsyncGenerator[PermissionService, None]:
    """Create and start a confirmation service."""
    await confirmation_service.start()
    yield confirmation_service
    await confirmation_service.stop()


class TestPermissionService:
    """Test cases for confirmation service."""

    async def test_request_permission_creates_request(
        self, started_service: PermissionService
    ) -> None:
        """Test that requesting confirmation creates a new request."""
        tool_name = "bash"
        input_params = {"command": "ls -la"}

        request_id = await started_service.request_permission(tool_name, input_params)

        assert request_id is not None
        assert len(request_id) > 0

        # Check request was stored
        request = await started_service.get_request(request_id)
        assert request is not None
        assert request.tool_name == tool_name
        assert request.input == input_params
        assert request.status == PermissionStatus.PENDING

    async def test_request_permission_validates_input(
        self, started_service: PermissionService
    ) -> None:
        """Test input validation for confirmation requests."""
        # Test empty tool name
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            await started_service.request_permission("", {"command": "test"})

        # Test whitespace-only tool name
        with pytest.raises(ValueError, match="Tool name cannot be empty"):
            await started_service.request_permission("   ", {"command": "test"})

        # Test None input
        with pytest.raises(ValueError, match="Input parameters cannot be None"):
            await started_service.request_permission("bash", None)  # type: ignore

    async def test_get_status_returns_correct_status(
        self, started_service: PermissionService
    ) -> None:
        """Test getting status of confirmation requests."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Check initial status
        status = await started_service.get_status(request_id)
        assert status == PermissionStatus.PENDING

        # Check non-existent request
        status = await started_service.get_status("non-existent-id")
        assert status is None

    async def test_resolve_confirmation_allowed(
        self, started_service: PermissionService
    ) -> None:
        """Test resolving a confirmation request as allowed."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Resolve as allowed
        success = await started_service.resolve(request_id, allowed=True)
        assert success is True

        # Check status updated
        status = await started_service.get_status(request_id)
        assert status == PermissionStatus.ALLOWED

    async def test_resolve_confirmation_denied(
        self, started_service: PermissionService
    ) -> None:
        """Test resolving a confirmation request as denied."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Resolve as denied
        success = await started_service.resolve(request_id, allowed=False)
        assert success is True

        # Check status updated
        status = await started_service.get_status(request_id)
        assert status == PermissionStatus.DENIED

    async def test_resolve_validates_input(
        self, started_service: PermissionService
    ) -> None:
        """Test input validation for resolve method."""
        # Test empty request ID
        with pytest.raises(ValueError, match="Request ID cannot be empty"):
            await started_service.resolve("", True)

        # Test whitespace-only request ID
        with pytest.raises(ValueError, match="Request ID cannot be empty"):
            await started_service.resolve("   ", True)

    async def test_resolve_non_existent_request(
        self, started_service: PermissionService
    ) -> None:
        """Test resolving a non-existent request returns False."""
        success = await started_service.resolve("non-existent-id", True)
        assert success is False

    async def test_resolve_already_resolved_request(
        self, started_service: PermissionService
    ) -> None:
        """Test resolving an already resolved request returns False."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # First resolution succeeds
        success = await started_service.resolve(request_id, True)
        assert success is True

        # Second resolution fails
        success = await started_service.resolve(request_id, False)
        assert success is False

    async def test_concurrent_resolutions(
        self, started_service: PermissionService
    ) -> None:
        """Test handling concurrent resolution attempts."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Attempt concurrent resolutions
        results = await asyncio.gather(
            started_service.resolve(request_id, True),
            started_service.resolve(request_id, False),
            return_exceptions=True,
        )

        # Only one should succeed
        successes = [r for r in results if r is True]
        assert len(successes) == 1

    async def test_event_subscription(self, started_service: PermissionService) -> None:
        """Test event subscription and emission."""
        # Subscribe to events
        queue = await started_service.subscribe_to_events()

        # Create a confirmation request
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Check we received the event
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "permission_request"
        assert event["request_id"] == request_id
        assert event["tool_name"] == "bash"

        # Resolve the request
        await started_service.resolve(request_id, True)

        # Check we received the resolution event
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "permission_resolved"
        assert event["request_id"] == request_id
        assert event["allowed"] is True

        # Unsubscribe
        await started_service.unsubscribe_from_events(queue)

    async def test_multiple_subscribers(
        self, started_service: PermissionService
    ) -> None:
        """Test multiple event subscribers receive events."""
        # Subscribe multiple queues
        queue1 = await started_service.subscribe_to_events()
        queue2 = await started_service.subscribe_to_events()

        # Create a request
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Both queues should receive the event
        event1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
        event2 = await asyncio.wait_for(queue2.get(), timeout=1.0)

        assert event1["request_id"] == request_id
        assert event2["request_id"] == request_id

        # Cleanup
        await started_service.unsubscribe_from_events(queue1)
        await started_service.unsubscribe_from_events(queue2)

    async def test_request_expiration(
        self, confirmation_service: PermissionService
    ) -> None:
        """Test that requests expire after timeout."""
        # Create service with very short timeout
        service = PermissionService(timeout_seconds=1)
        await service.start()

        try:
            request_id = await service.request_permission("bash", {"command": "test"})

            # Initially pending
            status = await service.get_status(request_id)
            assert status == PermissionStatus.PENDING

            # Wait for expiration
            await asyncio.sleep(1.1)

            # Should be expired now
            status = await service.get_status(request_id)
            assert status == PermissionStatus.EXPIRED

            # Cannot resolve expired request
            success = await service.resolve(request_id, True)
            assert success is False

        finally:
            await service.stop()

    async def test_wait_for_permission_allowed(
        self, started_service: PermissionService
    ) -> None:
        """Test waiting for a confirmation that gets allowed."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Resolve in background after delay
        async def resolve_later() -> None:
            await asyncio.sleep(0.1)
            await started_service.resolve(request_id, True)

        asyncio.create_task(resolve_later())

        # Wait for resolution
        status = await started_service.wait_for_permission(
            request_id, timeout_seconds=1
        )
        assert status == PermissionStatus.ALLOWED

    async def test_wait_for_permission_denied(
        self, started_service: PermissionService
    ) -> None:
        """Test waiting for a confirmation that gets denied."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Resolve in background after delay
        async def resolve_later() -> None:
            await asyncio.sleep(0.1)
            await started_service.resolve(request_id, False)

        asyncio.create_task(resolve_later())

        # Wait for resolution
        status = await started_service.wait_for_permission(
            request_id, timeout_seconds=1
        )
        assert status == PermissionStatus.DENIED

    async def test_wait_for_permission_timeout(
        self, started_service: PermissionService
    ) -> None:
        """Test waiting for a confirmation that times out."""
        request_id = await started_service.request_permission(
            "bash", {"command": "test"}
        )

        # Don't resolve - let it timeout
        with pytest.raises(asyncio.TimeoutError):
            await started_service.wait_for_permission(request_id, timeout_seconds=1)

    async def test_wait_for_non_existent_request(
        self, started_service: PermissionService
    ) -> None:
        """Test waiting for a non-existent request."""
        with pytest.raises(PermissionNotFoundError):
            await started_service.wait_for_permission("non-existent-id")

    async def test_cleanup_expired_requests(
        self, confirmation_service: PermissionService
    ) -> None:
        """Test that expired requests are cleaned up."""
        # Create service with very short cleanup time
        service = PermissionService(timeout_seconds=1)
        await service.start()

        try:
            # Subscribe to events to track expiration
            queue = await service.subscribe_to_events()

            request_id = await service.request_permission("bash", {"command": "test"})

            # Clear the creation event
            await asyncio.wait_for(queue.get(), timeout=1.0)

            # Wait for expiration checker to run (runs every 5 seconds)
            # But the request expires after 1 second
            await asyncio.sleep(6)

            # Should have received expiration event
            expired_event_received = False
            while not queue.empty():
                event = await queue.get()
                if event["type"] == "permission_expired":
                    expired_event_received = True
                    assert event["request_id"] == request_id

            assert expired_event_received

            # Request should be marked as expired
            status = await service.get_status(request_id)
            assert status == PermissionStatus.EXPIRED

        finally:
            await service.stop()

    async def test_get_permission_service_singleton(self) -> None:
        """Test that get_permission_service returns singleton."""
        service1 = get_permission_service()
        service2 = get_permission_service()
        assert service1 is service2

    async def test_get_pending_requests(self) -> None:
        """Test get_pending_requests returns only pending requests."""
        service = PermissionService()
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

    async def test_get_pending_requests_with_expired(self) -> None:
        """Test get_pending_requests updates expired requests."""
        service = PermissionService(timeout_seconds=0)
        await service.start()
        try:
            # Create a request that will immediately expire
            request_id = await service.request_permission("tool", {"param": "value"})

            # Wait a moment to ensure it's expired
            await asyncio.sleep(0.1)

            # Get pending requests
            pending = await service.get_pending_requests()

            # Should have no pending requests (expired ones are excluded)
            assert len(pending) == 0

            # Verify the request was marked as expired
            status = await service.get_status(request_id)
            assert status == PermissionStatus.EXPIRED
        finally:
            await service.stop()
