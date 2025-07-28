"""Tests for MCP permission checking functionality."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ccproxy.api.routes.mcp import PermissionCheckRequest, check_permission
from ccproxy.api.services.permission_service import (
    PermissionService,
)
from ccproxy.config.settings import Settings
from ccproxy.models.permissions import PermissionStatus
from ccproxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
)


@pytest.fixture
def mock_permission_service() -> Mock:
    """Create a mock permission service."""
    service = Mock(spec=PermissionService)
    service.request_permission = AsyncMock(return_value="test-permission-id")
    service.wait_for_permission = AsyncMock()
    service.get_status = AsyncMock()
    return service


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.security = Mock()
    settings.security.confirmation_timeout_seconds = 30
    return settings


class TestMCPPermissionCheck:
    """Test cases for MCP permission checking functionality."""

    async def test_check_permission_waits_and_allows(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission waits for permission and returns allow."""
        # Setup mock to return allowed status after waiting
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.ALLOWED
        )

        # Patch the service getter
        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "ls -la"},
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolAllowResponse)
            assert response.updated_input == {"command": "ls -la"}

            # Verify service was called
            mock_permission_service.request_permission.assert_called_once_with(
                tool_name="bash",
                input={"command": "ls -la"},
            )
            mock_permission_service.wait_for_permission.assert_called_once_with(
                "test-permission-id",
                timeout_seconds=30,
            )

    async def test_check_permission_with_permission_id_allowed(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with existing allowed permission."""
        # Setup mock to return allowed status
        mock_permission_service.get_status.return_value = PermissionStatus.ALLOWED

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request with permission ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "ls -la"},
                permission_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolAllowResponse)
            assert response.updated_input == {"command": "ls -la"}

            # Verify status was checked
            mock_permission_service.get_status.assert_called_once_with("existing-id")

    async def test_check_permission_with_permission_id_denied(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with existing denied permission."""
        # Setup mock to return denied status
        mock_permission_service.get_status.return_value = PermissionStatus.DENIED

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request with permission ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "rm -rf /"},
                permission_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "User denied the operation"

    async def test_check_permission_with_permission_id_expired(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with expired permission."""
        # Setup mock to return expired status
        mock_permission_service.get_status.return_value = PermissionStatus.EXPIRED

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request with permission ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "test"},
                permission_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "Permission request expired"

    async def test_check_permission_waits_and_denies(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission waits for permission and returns deny."""
        # Setup mock to return denied status after waiting
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.DENIED
        )

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request
            request = PermissionCheckRequest(
                tool_name="rm",
                input={"command": "rm -rf /"},
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert "User denied the operation" in response.message
            assert "denied" in response.message

    async def test_check_permission_timeout(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission handles timeout correctly."""
        # Setup mock to raise TimeoutError
        mock_permission_service.wait_for_permission.side_effect = TimeoutError()

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "test"},
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "Permission request timed out"

    async def test_check_permission_empty_tool_name(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that empty tool name is handled."""
        # Setup mock to return allowed status after waiting
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.ALLOWED
        )

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request with empty tool name - this is allowed by the model
            request = PermissionCheckRequest(
                tool_name="",
                input={"command": "test"},
            )

            # The service should handle this gracefully
            response = await check_permission(request, mock_settings)

            # Should still wait and return allowed
            assert isinstance(response, PermissionToolAllowResponse)

    async def test_check_permission_logs_appropriately(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that permission checks are logged."""
        # Setup mock to return allowed status after waiting
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.ALLOWED
        )

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            with patch("ccproxy.api.routes.mcp.logger") as mock_logger:
                # Create request
                request = PermissionCheckRequest(
                    tool_name="python",
                    input={"code": "print('hello')"},
                )

                # Call function
                await check_permission(request, mock_settings)

                # Verify logging
                mock_logger.info.assert_any_call(
                    "permission_check",
                    tool_name="python",
                    retry=False,
                )

                mock_logger.info.assert_any_call(
                    "permission_requires_authorization",
                    tool_name="python",
                )

                mock_logger.info.assert_any_call(
                    "permission_allowed_after_authorization",
                    tool_name="python",
                    permission_id="test-permission-id",
                )

    async def test_check_permission_with_tool_use_id(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test permission check with tool_use_id."""
        # Setup mock to return allowed status after waiting
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.ALLOWED
        )

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create request with tool_use_id
            request = PermissionCheckRequest(
                tool_name="file_write",
                input={"path": "/tmp/test.txt", "content": "test"},
                tool_use_id="tool-123",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolAllowResponse)

    async def test_check_permission_concurrent_requests(
        self,
        mock_permission_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test handling multiple concurrent permission requests."""
        # Setup mock to return different IDs
        call_count = 0

        async def mock_request_permission(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"permission-{call_count}"

        mock_permission_service.request_permission = mock_request_permission
        mock_permission_service.wait_for_permission.return_value = (
            PermissionStatus.ALLOWED
        )

        with patch("ccproxy.api.routes.mcp.get_permission_service") as mock_get_service:
            mock_get_service.return_value = mock_permission_service

            # Create multiple requests
            requests = [
                PermissionCheckRequest(
                    tool_name=f"tool-{i}",
                    input={"param": f"value-{i}"},
                )
                for i in range(5)
            ]

            # Call concurrently
            responses = await asyncio.gather(
                *[check_permission(req, mock_settings) for req in requests]
            )
            # Verify all got allow responses (since we mocked wait_for_permission to return ALLOWED)

            assert all(isinstance(r, PermissionToolAllowResponse) for r in responses)
