"""Protocol definition for confirmation handlers."""

from typing import Protocol

from ccproxy.api.services.confirmation_service import ConfirmationRequest


class ConfirmationHandlerProtocol(Protocol):
    """Protocol for confirmation request handlers.

    This protocol defines the interface that all confirmation handlers
    must implement to be compatible with the CLI confirmation system.
    """

    async def handle_confirmation(self, request: ConfirmationRequest) -> bool:
        """Handle a confirmation request.

        Args:
            request: The confirmation request to handle

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
