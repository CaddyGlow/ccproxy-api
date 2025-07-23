"""Confirmation service for handling permission requests without UI dependencies."""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from structlog import get_logger


logger = get_logger(__name__)


class ConfirmationStatus(Enum):
    PENDING = "pending"
    ALLOWED = "allowed"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ConfirmationRequest:
    """Represents a permission confirmation request."""

    id: str
    tool_name: str
    input: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    resolved_at: datetime | None = None

    @classmethod
    def create(
        cls, tool_name: str, input: dict[str, Any], timeout_seconds: int = 30
    ) -> "ConfirmationRequest":
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            input=input,
            created_at=now,
            expires_at=now + timedelta(seconds=timeout_seconds),
        )

    def is_expired(self) -> bool:
        if self.status != ConfirmationStatus.PENDING:
            return False
        return datetime.now() > self.expires_at

    def time_remaining(self) -> int:
        if self.status != ConfirmationStatus.PENDING:
            return 0
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))

    def resolve(self, allowed: bool) -> None:
        if self.status != ConfirmationStatus.PENDING:
            raise ValueError(f"Cannot resolve request in {self.status} status")

        self.status = (
            ConfirmationStatus.ALLOWED if allowed else ConfirmationStatus.DENIED
        )
        self.resolved_at = datetime.now()


class ConfirmationService:
    """Service for managing permission confirmation requests without UI dependencies."""

    def __init__(self, timeout_seconds: int = 30):
        self._timeout_seconds = timeout_seconds
        self._requests: dict[str, ConfirmationRequest] = {}
        self._confirmation_handler: (
            Callable[[ConfirmationRequest], Awaitable[bool]] | None
        ) = None
        self._expiry_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._event_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = asyncio.Lock()

    def set_confirmation_handler(
        self, handler: Callable[[ConfirmationRequest], Awaitable[bool]]
    ) -> None:
        """Set the handler function for confirmation requests.

        The handler should present the request to the user and return
        True if allowed, False if denied.

        Args:
            handler: Async function that handles user confirmation
        """
        self._confirmation_handler = handler

    async def start(self) -> None:
        if self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._expiry_checker())
            logger.info("confirmation_service_started")

    async def stop(self) -> None:
        self._shutdown = True
        if self._expiry_task:
            self._expiry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._expiry_task
            self._expiry_task = None
        logger.info("confirmation_service_stopped")

    async def request_confirmation(self, tool_name: str, input: dict[str, Any]) -> str:
        """Create a new confirmation request.

        Args:
            tool_name: Name of the tool requesting permission
            input: Input parameters for the tool

        Returns:
            Confirmation request ID
        """
        request = ConfirmationRequest.create(
            tool_name=tool_name,
            input=input,
            timeout_seconds=self._timeout_seconds,
        )

        self._requests[request.id] = request

        logger.info(
            "confirmation_request_created",
            request_id=request.id,
            tool_name=tool_name,
        )

        await self._emit_event(
            {
                "type": "confirmation_request",
                "request_id": request.id,
                "tool_name": request.tool_name,
                "input": request.input,
                "created_at": request.created_at.isoformat(),
                "expires_at": request.expires_at.isoformat(),
                "timeout_seconds": self._timeout_seconds,
            }
        )

        if self._confirmation_handler:
            asyncio.create_task(self._handle_confirmation(request))

        return request.id

    def get_status(self, request_id: str) -> ConfirmationStatus | None:
        """Get the status of a confirmation request.

        Args:
            request_id: ID of the confirmation request

        Returns:
            Status of the request or None if not found
        """
        request = self._requests.get(request_id)
        if not request:
            return None

        if request.is_expired():
            request.status = ConfirmationStatus.EXPIRED

        return request.status

    def get_request(self, request_id: str) -> ConfirmationRequest | None:
        """Get a confirmation request by ID.

        Args:
            request_id: ID of the confirmation request

        Returns:
            The request or None if not found
        """
        return self._requests.get(request_id)

    def resolve(self, request_id: str, allowed: bool) -> bool:
        """Manually resolve a confirmation request.

        Args:
            request_id: ID of the confirmation request
            allowed: Whether to allow or deny the request

        Returns:
            True if resolved successfully, False if not found or already resolved
        """
        request = self._requests.get(request_id)
        if not request or request.status != ConfirmationStatus.PENDING:
            return False

        try:
            request.resolve(allowed)
            logger.info(
                "confirmation_request_resolved",
                request_id=request_id,
                tool_name=request.tool_name,
                allowed=allowed,
            )

            # Emit resolution event
            asyncio.create_task(
                self._emit_event(
                    {
                        "type": "confirmation_resolved",
                        "request_id": request_id,
                        "allowed": allowed,
                        "resolved_at": request.resolved_at.isoformat()
                        if request.resolved_at
                        else None,
                    }
                )
            )

            return True
        except ValueError:
            return False

    async def _handle_confirmation(self, request: ConfirmationRequest) -> None:
        if not self._confirmation_handler:
            return

        try:
            timeout = request.time_remaining()

            result = await asyncio.wait_for(
                self._confirmation_handler(request),
                timeout=timeout,
            )

            if request.status == ConfirmationStatus.PENDING:
                request.resolve(result)

        except TimeoutError:
            if request.status == ConfirmationStatus.PENDING:
                request.status = ConfirmationStatus.EXPIRED
                logger.warning(
                    "confirmation_handler_timeout",
                    request_id=request.id,
                    tool_name=request.tool_name,
                )
        except Exception as e:
            if request.status == ConfirmationStatus.PENDING:
                request.resolve(False)
            logger.error(
                "confirmation_handler_error",
                request_id=request.id,
                error=str(e),
                exc_info=True,
            )

    async def _expiry_checker(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.sleep(5)

                now = datetime.now()
                expired_ids = []

                for req_id, req in self._requests.items():
                    if req.is_expired() and req.status == ConfirmationStatus.PENDING:
                        req.status = ConfirmationStatus.EXPIRED

                    if self._should_cleanup_request(req, now):
                        expired_ids.append(req_id)

                for req_id in expired_ids:
                    del self._requests[req_id]

                if expired_ids:
                    logger.info(
                        "cleaned_expired_requests",
                        count=len(expired_ids),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "expiry_checker_error",
                    error=str(e),
                    exc_info=True,
                )

    def _should_cleanup_request(
        self, request: ConfirmationRequest, now: datetime
    ) -> bool:
        """Check if a resolved request should be cleaned up."""
        if request.status == ConfirmationStatus.PENDING:
            return False

        cleanup_after = timedelta(minutes=5)

        if request.resolved_at:
            return (now - request.resolved_at) > cleanup_after

        if request.status == ConfirmationStatus.EXPIRED:
            return (now - request.expires_at) > cleanup_after

        return False

    async def subscribe_to_events(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to confirmation events.

        Returns:
            An async queue that will receive events
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        async with self._lock:
            self._event_queues.append(queue)
        return queue

    async def unsubscribe_from_events(
        self, queue: asyncio.Queue[dict[str, Any]]
    ) -> None:
        """Unsubscribe from confirmation events.

        Args:
            queue: The queue to unsubscribe
        """
        async with self._lock:
            if queue in self._event_queues:
                self._event_queues.remove(queue)

    async def wait_for_confirmation(
        self, request_id: str, timeout_seconds: int | None = None
    ) -> ConfirmationStatus:
        """Wait for a confirmation request to be resolved.

        This method blocks until the confirmation is resolved (allowed/denied/expired)
        or the timeout is reached.

        Args:
            request_id: ID of the confirmation request to wait for
            timeout_seconds: Optional timeout in seconds. If None, uses request expiration time

        Returns:
            The final status of the confirmation request

        Raises:
            asyncio.TimeoutError: If timeout is reached before resolution
            ValueError: If request ID is not found
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Confirmation request {request_id} not found")

        if request.status != ConfirmationStatus.PENDING:
            return request.status

        if timeout_seconds is None:
            timeout_seconds = request.time_remaining()

        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.1

        while True:
            current_status = self.get_status(request_id)
            if current_status is None:
                return ConfirmationStatus.EXPIRED
            if current_status != ConfirmationStatus.PENDING:
                return current_status

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                logger.warning(
                    "confirmation_wait_timeout",
                    request_id=request_id,
                    elapsed_seconds=elapsed,
                )
                if request.status == ConfirmationStatus.PENDING:
                    request.status = ConfirmationStatus.EXPIRED
                raise TimeoutError(f"Confirmation wait timeout after {elapsed:.1f}s")

            await asyncio.sleep(poll_interval)

    async def _emit_event(self, event: dict[str, Any]) -> None:
        """Emit an event to all subscribers.

        Args:
            event: The event data to emit
        """
        async with self._lock:
            queues = list(self._event_queues)

        if not queues:
            return

        for queue in queues:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)


# Global instance
_confirmation_service: ConfirmationService | None = None


def get_confirmation_service() -> ConfirmationService:
    """Get the global confirmation service instance."""
    global _confirmation_service
    if _confirmation_service is None:
        _confirmation_service = ConfirmationService()
    return _confirmation_service
