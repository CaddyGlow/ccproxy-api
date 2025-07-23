"""Confirmation service for handling permission requests without UI dependencies."""

import asyncio
import contextlib
from datetime import datetime, timedelta
from typing import Any

from structlog import get_logger

from ccproxy.core.errors import (
    ConfirmationAlreadyResolvedError,
    ConfirmationExpiredError,
    ConfirmationNotFoundError,
)
from ccproxy.models.confirmations import (
    ConfirmationEvent,
    ConfirmationRequest,
    ConfirmationStatus,
)


logger = get_logger(__name__)


class ConfirmationService:
    """Service for managing permission confirmation requests without UI dependencies."""

    def __init__(self, timeout_seconds: int = 30):
        self._timeout_seconds = timeout_seconds
        self._requests: dict[str, ConfirmationRequest] = {}
        self._expiry_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._event_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = asyncio.Lock()

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

    async def request_confirmation(self, tool_name: str, input: dict[str, str]) -> str:
        """Create a new confirmation request.

        Args:
            tool_name: Name of the tool requesting permission
            input: Input parameters for the tool

        Returns:
            Confirmation request ID

        Raises:
            ValueError: If tool_name is empty or input is None
        """
        # Input validation
        if not tool_name or not tool_name.strip():
            raise ValueError("Tool name cannot be empty")
        if input is None:
            raise ValueError("Input parameters cannot be None")

        # Sanitize input - ensure all values are strings
        sanitized_input = {k: str(v) for k, v in input.items()}

        now = datetime.utcnow()
        request = ConfirmationRequest(
            tool_name=tool_name.strip(),
            input=sanitized_input,
            created_at=now,
            expires_at=now + timedelta(seconds=self._timeout_seconds),
        )

        async with self._lock:
            self._requests[request.id] = request

        logger.info(
            "confirmation_request_created",
            request_id=request.id,
            tool_name=tool_name,
        )

        event = ConfirmationEvent(
            type="confirmation_request",
            request_id=request.id,
            tool_name=request.tool_name,
            input=request.input,
            created_at=request.created_at.isoformat(),
            expires_at=request.expires_at.isoformat(),
            timeout_seconds=self._timeout_seconds,
        )
        await self._emit_event(event.model_dump())

        return request.id

    async def get_status(self, request_id: str) -> ConfirmationStatus | None:
        """Get the status of a confirmation request.

        Args:
            request_id: ID of the confirmation request

        Returns:
            Status of the request or None if not found
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return None

            if request.is_expired():
                request.status = ConfirmationStatus.EXPIRED

            return request.status

    async def get_request(self, request_id: str) -> ConfirmationRequest | None:
        """Get a confirmation request by ID.

        Args:
            request_id: ID of the confirmation request

        Returns:
            The request or None if not found
        """
        async with self._lock:
            return self._requests.get(request_id)

    async def resolve(self, request_id: str, allowed: bool) -> bool:
        """Manually resolve a confirmation request.

        Args:
            request_id: ID of the confirmation request
            allowed: Whether to allow or deny the request

        Returns:
            True if resolved successfully, False if not found or already resolved

        Raises:
            ValueError: If request_id is empty
        """
        # Input validation
        if not request_id or not request_id.strip():
            raise ValueError("Request ID cannot be empty")

        async with self._lock:
            request = self._requests.get(request_id.strip())
            if not request or request.status != ConfirmationStatus.PENDING:
                return False

            try:
                request.resolve(allowed)
            except ValueError:
                return False

        logger.info(
            "confirmation_request_resolved",
            request_id=request_id,
            tool_name=request.tool_name,
            allowed=allowed,
        )

        # Emit resolution event
        event = ConfirmationEvent(
            type="confirmation_resolved",
            request_id=request_id,
            allowed=allowed,
            resolved_at=request.resolved_at.isoformat()
            if request.resolved_at
            else None,
        )
        await self._emit_event(event.model_dump())

        return True

    async def _expiry_checker(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.sleep(5)

                now = datetime.utcnow()
                expired_ids = []
                expired_events = []

                async with self._lock:
                    for req_id, req in self._requests.items():
                        if (
                            req.is_expired()
                            and req.status == ConfirmationStatus.PENDING
                        ):
                            req.status = ConfirmationStatus.EXPIRED
                            event = ConfirmationEvent(
                                type="confirmation_expired",
                                request_id=req_id,
                                expired_at=now.isoformat(),
                            )
                            expired_events.append(event.model_dump())

                        if self._should_cleanup_request(req, now):
                            expired_ids.append(req_id)

                    for req_id in expired_ids:
                        del self._requests[req_id]

                # Emit expired events outside the lock
                for event_data in expired_events:
                    await self._emit_event(event_data)

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
            ConfirmationNotFoundError: If request ID is not found
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                raise ConfirmationNotFoundError(request_id)

            if request.status != ConfirmationStatus.PENDING:
                return request.status

        if timeout_seconds is None:
            timeout_seconds = request.time_remaining()

        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.1

        while True:
            current_status = await self.get_status(request_id)
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
                async with self._lock:
                    if request_id in self._requests:
                        req = self._requests[request_id]
                        if req.status == ConfirmationStatus.PENDING:
                            req.status = ConfirmationStatus.EXPIRED
                raise TimeoutError(f"Confirmation wait timeout after {elapsed:.1f}s")

            await asyncio.sleep(poll_interval)


# Global instance
_confirmation_service: ConfirmationService | None = None


def get_confirmation_service() -> ConfirmationService:
    """Get the global confirmation service instance."""
    global _confirmation_service
    if _confirmation_service is None:
        _confirmation_service = ConfirmationService()
    return _confirmation_service


__all__ = [
    "ConfirmationService",
    "ConfirmationRequest",
    "get_confirmation_service",
]
