"""Tests for MCP permission checking functionality."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ccproxy.api.routes.mcp import PermissionCheckRequest, check_permission
from ccproxy.api.services.confirmation_service import (
    ConfirmationService,
    get_confirmation_service,
)
from ccproxy.config.settings import Settings, get_settings
from ccproxy.models.confirmations import ConfirmationStatus
from ccproxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
    PermissionToolPendingResponse,
)


@pytest.fixture
def mock_confirmation_service() -> Mock:
    """Create a mock confirmation service."""
    service = Mock(spec=ConfirmationService)
    service.request_confirmation = AsyncMock(return_value="test-confirmation-id")
    service.wait_for_confirmation = AsyncMock()
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
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission waits for confirmation and returns allow."""
        # Setup mock to return allowed status after waiting
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.ALLOWED
        )

        # Patch the service getter
        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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
            mock_confirmation_service.request_confirmation.assert_called_once_with(
                tool_name="bash",
                input={"command": "ls -la"},
            )
            mock_confirmation_service.wait_for_confirmation.assert_called_once_with(
                "test-confirmation-id",
                timeout_seconds=30,
            )

    async def test_check_permission_with_confirmation_id_allowed(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with existing allowed confirmation."""
        # Setup mock to return allowed status
        mock_confirmation_service.get_status.return_value = ConfirmationStatus.ALLOWED

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Create request with confirmation ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "ls -la"},
                confirmation_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolAllowResponse)
            assert response.updated_input == {"command": "ls -la"}

            # Verify status was checked
            mock_confirmation_service.get_status.assert_called_once_with("existing-id")

    async def test_check_permission_with_confirmation_id_denied(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with existing denied confirmation."""
        # Setup mock to return denied status
        mock_confirmation_service.get_status.return_value = ConfirmationStatus.DENIED

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Create request with confirmation ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "rm -rf /"},
                confirmation_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "User denied the operation"

    async def test_check_permission_with_confirmation_id_expired(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test checking permission with expired confirmation."""
        # Setup mock to return expired status
        mock_confirmation_service.get_status.return_value = ConfirmationStatus.EXPIRED

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Create request with confirmation ID
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "test"},
                confirmation_id="existing-id",
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "Confirmation request expired"

    async def test_check_permission_waits_and_denies(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission waits for confirmation and returns deny."""
        # Setup mock to return denied status after waiting
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.DENIED
        )

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that check-permission handles timeout correctly."""
        # Setup mock to raise TimeoutError
        mock_confirmation_service.wait_for_confirmation.side_effect = TimeoutError()

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

            # Create request
            request = PermissionCheckRequest(
                tool_name="bash",
                input={"command": "test"},
            )

            # Call function
            response = await check_permission(request, mock_settings)

            # Verify response
            assert isinstance(response, PermissionToolDenyResponse)
            assert response.message == "Confirmation request timed out"

    async def test_check_permission_empty_tool_name(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that empty tool name is handled."""
        # Setup mock to return allowed status after waiting
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.ALLOWED
        )

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test that permission checks are logged."""
        # Setup mock to return allowed status after waiting
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.ALLOWED
        )

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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
                    "permission_requires_confirmation",
                    tool_name="python",
                )

                mock_logger.info.assert_any_call(
                    "permission_allowed_after_confirmation",
                    tool_name="python",
                    confirmation_id="test-confirmation-id",
                )

    async def test_check_permission_with_tool_use_id(
        self,
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test permission check with tool_use_id."""
        # Setup mock to return allowed status after waiting
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.ALLOWED
        )

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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
        mock_confirmation_service: Mock,
        mock_settings: Settings,
    ) -> None:
        """Test handling multiple concurrent permission requests."""
        # Setup mock to return different IDs
        call_count = 0

        async def mock_request_confirmation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"confirmation-{call_count}"

        mock_confirmation_service.request_confirmation = mock_request_confirmation
        mock_confirmation_service.wait_for_confirmation.return_value = (
            ConfirmationStatus.ALLOWED
        )

        with patch(
            "ccproxy.api.routes.mcp.get_confirmation_service"
        ) as mock_get_service:
            mock_get_service.return_value = mock_confirmation_service

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

            # Verify all got allow responses (since we mocked wait_for_confirmation to return ALLOWED)
            assert all(isinstance(r, PermissionToolAllowResponse) for r in responses)
