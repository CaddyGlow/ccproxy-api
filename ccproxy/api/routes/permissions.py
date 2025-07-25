"""API routes for permission request handling via SSE and REST."""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from structlog import get_logger

from ccproxy.api.dependencies import SettingsDep
from ccproxy.api.services.permission_service import get_permission_service
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.core.errors import (
    PermissionAlreadyResolvedError,
    PermissionNotFoundError,
)
from ccproxy.models.permissions import EventType, PermissionEvent, PermissionStatus


logger = get_logger(__name__)


router = APIRouter(tags=["permissions"])


class PermissionResponse(BaseModel):
    """Response to a permission request."""

    allowed: bool


class PermissionRequestInfo(BaseModel):
    """Information about a permission request."""

    request_id: str
    tool_name: str
    input: dict[str, str]
    status: str
    created_at: str
    expires_at: str
    time_remaining: int


async def event_generator(
    request: Request,
) -> AsyncGenerator[dict[str, str], None]:
    """Generate SSE events for permission requests.

    Args:
        request: The FastAPI request object

    Yields:
        Dict with event data for SSE
    """
    service = get_permission_service()
    queue = await service.subscribe_to_events()

    try:
        yield {
            "event": "ping",
            "data": json.dumps({"message": "Connected to permission stream"}),
        }

        # Send all pending permission requests to the newly connected client
        pending_requests = await service.get_pending_requests()
        for pending_req in pending_requests:
            event = PermissionEvent(
                type=EventType.PERMISSION_REQUEST,
                request_id=pending_req.id,
                tool_name=pending_req.tool_name,
                input=pending_req.input,
                created_at=pending_req.created_at.isoformat(),
                expires_at=pending_req.expires_at.isoformat(),
                timeout_seconds=int(
                    (pending_req.expires_at - pending_req.created_at).total_seconds()
                ),
            )
            yield {
                "event": EventType.PERMISSION_REQUEST.value,
                "data": json.dumps(event.model_dump(mode="json")),
            }

        while not await request.is_disconnected():
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=30.0)

                yield {
                    "event": event_data.get("type", "message"),
                    "data": json.dumps(event_data),
                }

            except TimeoutError:
                yield {
                    "event": "ping",
                    "data": json.dumps({"message": "keepalive"}),
                }

    except asyncio.CancelledError:
        pass
    finally:
        await service.unsubscribe_from_events(queue)


@router.get("/stream")
async def stream_permissions(
    request: Request,
    settings: SettingsDep,
    auth: ConditionalAuthDep,
) -> EventSourceResponse:
    """Stream permission requests via Server-Sent Events.

    This endpoint streams new permission requests as they are created,
    allowing external tools to handle user permissions.

    Returns:
        EventSourceResponse streaming permission events
    """
    return EventSourceResponse(
        event_generator(request),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{permission_id}")
async def get_permission(
    permission_id: str,
    settings: SettingsDep,
    auth: ConditionalAuthDep,
) -> PermissionRequestInfo:
    """Get information about a specific permission request.

    Args:
        permission_id: ID of the permission request

    Returns:
        Information about the permission request

    Raises:
        HTTPException: If request not found
    """
    service = get_permission_service()
    try:
        request = await service.get_request(permission_id)
        if not request:
            raise PermissionNotFoundError(permission_id)
    except PermissionNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="Permission request not found"
        ) from e

    return PermissionRequestInfo(
        request_id=request.id,
        tool_name=request.tool_name,
        input=request.input,
        status=request.status.value,
        created_at=request.created_at.isoformat(),
        expires_at=request.expires_at.isoformat(),
        time_remaining=request.time_remaining(),
    )


@router.post("/{permission_id}/respond")
async def respond_to_permission(
    permission_id: str,
    response: PermissionResponse,
    settings: SettingsDep,
    auth: ConditionalAuthDep,
) -> dict[str, str | bool]:
    """Submit a response to a permission request.

    Args:
        permission_id: ID of the permission request
        response: The allow/deny response

    Returns:
        Success response

    Raises:
        HTTPException: If request not found or already resolved
    """
    service = get_permission_service()
    status = await service.get_status(permission_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Permission request not found")

    if status != PermissionStatus.PENDING:
        try:
            raise PermissionAlreadyResolvedError(permission_id, status.value)
        except PermissionAlreadyResolvedError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=e.message,
            ) from e

    success = await service.resolve(permission_id, response.allowed)

    if not success:
        raise HTTPException(
            status_code=409, detail="Failed to resolve permission request"
        )

    logger.info(
        "permission_resolved_via_api",
        permission_id=permission_id,
        allowed=response.allowed,
    )

    return {
        "status": "success",
        "permission_id": permission_id,
        "allowed": response.allowed,
    }
