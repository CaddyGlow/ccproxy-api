"""Protocol definition for confirmation handlers."""

from typing import Protocol

from ..models import PermissionRequest


class ConfirmationHandlerProtocol(Protocol):
    """Protocol for confirmation request handlers.

    This protocol defines the interface that all confirmation handlers
    must implement to be compatible with the CLI confirmation system.
    """

    async def handle_permission(self, request: PermissionRequest) -> bool:
        """Handle a permission request.

        Args:
            request: The permission request to handle

        Returns:
            bool: True if the user confirmed, False otherwise
        """
        ...

    def cancel_confirmation(self, request_id: str, reason: str = "cancelled") -> None:
        """Cancel an ongoing confirmation request.

        Args:
            request_id: The ID of the request to cancel
            reason: The reason for cancellation
        """
        ...
