"""Permission service for handling permission requests without UI dependencies."""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from ccproxy.core.async_task_manager import AsyncTaskManager, create_managed_task
from ccproxy.core.errors import (
    PermissionNotFoundError,
)
from ccproxy.core.logging import get_plugin_logger

from .models import (
    EventType,
    PermissionEvent,
    PermissionRequest,
    PermissionStatus,
)


if TYPE_CHECKING:
    from ccproxy.services.container import ServiceContainer


logger = get_plugin_logger()


class PermissionService:
    """Service for managing permission requests without UI dependencies."""

    def __init__(self, timeout_seconds: int = 30):
        self._timeout_seconds = timeout_seconds
        self._requests: dict[str, PermissionRequest] = {}
        self._expiry_task: asyncio.Task[None] | None = None
        self._shutdown = False
        self._event_queues: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        container: "ServiceContainer | None" = None,
        task_manager: AsyncTaskManager | None = None,
    ) -> None:
        if self._expiry_task is not None:
            return

        self._shutdown = False

        try:
            self._expiry_task = await create_managed_task(
                self._expiry_checker(),
                name="permission_expiry_checker",
                creator="PermissionService",
                container=container,
                task_manager=task_manager,
            )
        except RuntimeError as exc:
            if not self._should_fallback_to_unmanaged_task(exc):
                raise

            logger.warning(
                "permission_service_task_manager_unavailable",
                error=str(exc),
            )
            self._expiry_task = asyncio.create_task(
                self._expiry_checker(), name="permission_expiry_checker"
            )

        logger.debug("permission_service_started")

    async def stop(self) -> None:
        self._shutdown = True
        if self._expiry_task:
            self._expiry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._expiry_task
            self._expiry_task = None
        logger.debug("permission_service_stopped")

    async def request_permission(self, tool_name: str, input: dict[str, str]) -> str:
        """Create a new permission request.

        Args:
            tool_name: Name of the tool requesting permission
            input: Input parameters for the tool

        Returns:
            Permission request ID

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

        now = datetime.now(UTC)
        request = PermissionRequest(
            tool_name=tool_name.strip(),
            input=sanitized_input,
            created_at=now,
            expires_at=now + timedelta(seconds=self._timeout_seconds),
        )

        async with self._lock:
            self._requests[request.id] = request

        logger.info(
            "permission_request_created",
            request_id=request.id,
            tool_name=tool_name,
        )

        event = PermissionEvent(
            type=EventType.PERMISSION_REQUEST,
            request_id=request.id,
            tool_name=request.tool_name,
            input=request.input,
            created_at=request.created_at.isoformat(),
            expires_at=request.expires_at.isoformat(),
            timeout_seconds=self._timeout_seconds,
        )
        await self._emit_event(event.model_dump(mode="json"))

        return request.id

    async def get_status(self, request_id: str) -> PermissionStatus | None:
        """Get the status of a permission request.

        Args:
            request_id: ID of the permission request

        Returns:
            Status of the request or None if not found
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return None

            if request.is_expired():
                request.status = PermissionStatus.EXPIRED

            return request.status

    async def get_request(self, request_id: str) -> PermissionRequest | None:
        """Get a permission request by ID.

        Args:
            request_id: ID of the permission request

        Returns:
            The request or None if not found
        """
        async with self._lock:
            return self._requests.get(request_id)

    async def resolve(self, request_id: str, allowed: bool) -> bool:
        """Manually resolve a permission request.

        Args:
            request_id: ID of the permission request
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
            if not request or request.status != PermissionStatus.PENDING:
                return False

            try:
                request.resolve(allowed)
            except ValueError:
                return False

        logger.info(
            "permission_request_resolved",
            request_id=request_id,
            tool_name=request.tool_name,
            allowed=allowed,
        )

        # Emit resolution event
        event = PermissionEvent(
            type=EventType.PERMISSION_RESOLVED,
            request_id=request_id,
            allowed=allowed,
            resolved_at=request.resolved_at.isoformat()
            if request.resolved_at
            else None,
        )
        await self._emit_event(event.model_dump(mode="json"))

        return True

    async def _expiry_checker(self) -> None:
        while not self._shutdown:
            try:
                await asyncio.sleep(self._get_expiry_poll_interval())

                now = datetime.now(UTC)
                expired_ids = []
                expired_events = []

                async with self._lock:
                    for req_id, req in self._requests.items():
                        if req.is_expired() and req.status == PermissionStatus.PENDING:
                            req.status = PermissionStatus.EXPIRED
                            # Signal waiting coroutines that the request is resolved (expired)
                            req._resolved_event.set()
                            event = PermissionEvent(
                                type=EventType.PERMISSION_EXPIRED,
                                request_id=req_id,
                                expired_at=now.isoformat(),
                            )
                            expired_events.append(event.model_dump(mode="json"))

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
                    exc_info=e,
                )

    def _should_cleanup_request(
        self, request: PermissionRequest, now: datetime
    ) -> bool:
        """Check if a resolved request should be cleaned up."""
        if request.status == PermissionStatus.PENDING:
            return False

        cleanup_after = timedelta(minutes=5)

        if request.resolved_at:
            return (now - request.resolved_at) > cleanup_after

        if request.status == PermissionStatus.EXPIRED:
            return (now - request.expires_at) > cleanup_after

        return False

    def _get_expiry_poll_interval(self) -> float:
        """Determine how frequently to poll for expired requests."""

        timeout = max(self._timeout_seconds, 0)
        if timeout == 0:
            return 0.5

        return max(0.5, min(5.0, timeout / 2))

    @staticmethod
    def _should_fallback_to_unmanaged_task(exc: RuntimeError) -> bool:
        message = str(exc)
        return any(
            hint in message
            for hint in (
                "Task manager is not started",
                "ServiceContainer is not available",
                "AsyncTaskManager is not registered",
            )
        )

    async def subscribe_to_events(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to permission events.

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
        """Unsubscribe from permission events.

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

    async def get_pending_requests(self) -> list[PermissionRequest]:
        """Get all pending permission requests.

        Returns:
            List of pending requests
        """
        async with self._lock:
            pending = []
            now = datetime.now(UTC)
            for request in self._requests.values():
                if request.is_expired():
                    request.status = PermissionStatus.EXPIRED
                elif request.status == PermissionStatus.PENDING:
                    pending.append(request)
            return pending

    async def wait_for_permission(
        self, request_id: str, timeout_seconds: int | None = None
    ) -> PermissionStatus:
        """Wait for a permission request to be resolved.

        This method efficiently blocks until the permission is resolved (allowed/denied/expired)
        or the timeout is reached using an event-driven approach.

        Args:
            request_id: ID of the permission request to wait for
            timeout_seconds: Optional timeout in seconds. If None, uses request expiration time

        Returns:
            The final status of the permission request

        Raises:
            asyncio.TimeoutError: If timeout is reached before resolution
            PermissionNotFoundError: If request ID is not found
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                raise PermissionNotFoundError(request_id)

            if request.status != PermissionStatus.PENDING:
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
                "permission_wait_timeout",
                request_id=request_id,
                timeout_seconds=timeout_seconds,
            )
            # Ensure status is updated to EXPIRED on timeout
            async with self._lock:
                if request.status == PermissionStatus.PENDING:
                    request.status = PermissionStatus.EXPIRED
                    request._resolved_event.set()  # Signal that it's resolved (as expired)
            raise TimeoutError(
                f"Confirmation wait timeout after {timeout_seconds:.1f}s"
            ) from e

        # The event is set, so the status is resolved
        return await self.get_status(request_id) or PermissionStatus.EXPIRED


# Global instance
_permission_service: PermissionService | None = None


def get_permission_service() -> PermissionService:
    """Get the global permission service instance."""
    global _permission_service
    if _permission_service is None:
        _permission_service = PermissionService()
    return _permission_service


__all__ = [
    "PermissionService",
    "PermissionRequest",
    "get_permission_service",
]
