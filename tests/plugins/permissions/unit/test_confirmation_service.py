"""Unit tests for permission service models and basic functionality."""

from unittest.mock import Mock, patch

import pytest

from ccproxy.plugins.permissions.models import PermissionRequest, PermissionStatus
from ccproxy.plugins.permissions.service import (
    PermissionService,
    get_permission_service,
)


@pytest.fixture
def mock_create_managed_task():
    """Mock the create_managed_task function to avoid task manager dependency."""
    with patch("ccproxy.plugins.permissions.service.create_managed_task") as mock:
        # Return a mock task that can be cancelled
        mock_task = Mock()
        mock_task.cancel = Mock()
        mock.return_value = mock_task
        yield mock


@pytest.fixture
def confirmation_service(mock_create_managed_task) -> PermissionService:
    """Create a test confirmation service."""
    service = PermissionService(timeout_seconds=30)
    return service


class TestPermissionService:
    """Test cases for permission service."""

    def test_service_creation(self, confirmation_service: PermissionService) -> None:
        """Test that service can be created."""
        assert confirmation_service is not None
        assert confirmation_service._timeout_seconds == 30
        assert len(confirmation_service._requests) == 0
        assert confirmation_service._shutdown is False

    def test_get_permission_service_singleton(self) -> None:
        """Test that get_permission_service returns singleton."""
        service1 = get_permission_service()
        service2 = get_permission_service()
        assert service1 is service2


class TestPermissionRequest:
    """Test cases for PermissionRequest model."""

    def test_permission_request_creation(self) -> None:
        """Test creating a permission request."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "ls -la"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        assert request.tool_name == "bash"
        assert request.input == {"command": "ls -la"}
        assert request.status == PermissionStatus.PENDING
        assert request.created_at == now
        assert request.expires_at == now + timedelta(seconds=30)
        assert request.resolved_at is None
        assert len(request.id) > 0

    def test_permission_request_resolve_allowed(self) -> None:
        """Test resolving a request as allowed."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        # Initially pending
        assert request.status == PermissionStatus.PENDING
        assert request.resolved_at is None

        # Resolve as allowed
        request.resolve(True)

        assert request.status == PermissionStatus.ALLOWED
        assert request.resolved_at is not None

    def test_permission_request_resolve_denied(self) -> None:
        """Test resolving a request as denied."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        # Resolve as denied
        request.resolve(False)

        assert request.status == PermissionStatus.DENIED
        assert request.resolved_at is not None

    def test_permission_request_cannot_resolve_twice(self) -> None:
        """Test that a request cannot be resolved twice."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        # First resolution succeeds
        request.resolve(True)
        assert request.status == PermissionStatus.ALLOWED

        # Second resolution should raise ValueError
        with pytest.raises(ValueError, match="Cannot resolve request in"):
            request.resolve(False)

    def test_permission_request_is_expired(self) -> None:
        """Test checking if a request is expired."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)

        # Create expired request
        expired_request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now - timedelta(seconds=60),
            expires_at=now - timedelta(seconds=30),
        )

        # Create non-expired request
        active_request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        assert expired_request.is_expired() is True
        assert active_request.is_expired() is False

    def test_permission_request_time_remaining(self) -> None:
        """Test calculating time remaining."""
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)

        # Create request expiring in 30 seconds
        request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now,
            expires_at=now + timedelta(seconds=30),
        )

        time_remaining = request.time_remaining()
        # Should be approximately 30 seconds (allow for small timing differences)
        assert 29 <= time_remaining <= 30

        # Expired request should return 0
        expired_request = PermissionRequest(
            tool_name="bash",
            input={"command": "test"},
            created_at=now - timedelta(seconds=60),
            expires_at=now - timedelta(seconds=30),
        )

        assert expired_request.time_remaining() == 0
