"""Confirmation service for handling permission requests without UI dependencies."""

import asyncio
import contextlib
from datetime import datetime, timedelta
from typing import cast

from structlog import get_logger

from ccproxy.core.errors import (
    ConfirmationAlreadyResolvedError,
    ConfirmationExpiredError,
    ConfirmationNotFoundError,
)
from ccproxy.models.confirmations import (
    ConfirmationEvent,
    ConfirmationEventDict,
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
        self._event_queues: list[asyncio.Queue[ConfirmationEventDict]] = []
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
        await self._emit_event(cast(ConfirmationEventDict, event.model_dump()))

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
        await self._emit_event(cast(ConfirmationEventDict, event.model_dump()))

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
                            # Signal waiting coroutines that the request is resolved (expired)
                            req._resolved_event.set()
                            event = ConfirmationEvent(
                                type="confirmation_expired",
                                request_id=req_id,
                                expired_at=now.isoformat(),
                            )
                            expired_events.append(
                                cast(ConfirmationEventDict, event.model_dump())
                            )

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

    async def subscribe_to_events(self) -> asyncio.Queue[ConfirmationEventDict]:
        """Subscribe to confirmation events.

        Returns:
            An async queue that will receive events
        """
        queue: asyncio.Queue[ConfirmationEventDict] = asyncio.Queue()
        async with self._lock:
            self._event_queues.append(queue)
        return queue

    async def unsubscribe_from_events(
        self, queue: asyncio.Queue[ConfirmationEventDict]
    ) -> None:
        """Unsubscribe from confirmation events.

        Args:
            queue: The queue to unsubscribe
        """
        async with self._lock:
            if queue in self._event_queues:
                self._event_queues.remove(queue)

    async def _emit_event(self, event: ConfirmationEventDict) -> None:
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

    async def get_pending_requests(self) -> list[ConfirmationRequest]:
        """Get all pending confirmation requests.

        Returns:
            List of pending requests
        """
        async with self._lock:
            pending = []
            now = datetime.utcnow()
            for request in self._requests.values():
                if request.is_expired():
                    request.status = ConfirmationStatus.EXPIRED
                elif request.status == ConfirmationStatus.PENDING:
                    pending.append(request)
            return pending

    async def wait_for_confirmation(
        self, request_id: str, timeout_seconds: int | None = None
    ) -> ConfirmationStatus:
        """Wait for a confirmation request to be resolved.

        This method efficiently blocks until the confirmation is resolved (allowed/denied/expired)
        or the timeout is reached using an event-driven approach.

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

        try:
            # Efficiently wait for the event to be set
            await asyncio.wait_for(
                request._resolved_event.wait(), timeout=timeout_seconds
            )
        except TimeoutError as e:
            logger.warning(
                "confirmation_wait_timeout",
                request_id=request_id,
                timeout_seconds=timeout_seconds,
            )
            # Ensure status is updated to EXPIRED on timeout
            async with self._lock:
                if request.status == ConfirmationStatus.PENDING:
                    request.status = ConfirmationStatus.EXPIRED
                    request._resolved_event.set()  # Signal that it's resolved (as expired)
            raise TimeoutError(
                f"Confirmation wait timeout after {timeout_seconds:.1f}s"
            ) from e

        # The event is set, so the status is resolved
        return await self.get_status(request_id) or ConfirmationStatus.EXPIRED


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
