"""API routes for confirmation request handling via SSE and REST."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from structlog import get_logger

from ccproxy.api.services.confirmation_service import get_confirmation_service
from ccproxy.auth.conditional import ConditionalAuthDep
from ccproxy.config.settings import Settings, get_settings
from ccproxy.core.errors import (
    ConfirmationAlreadyResolvedError,
    ConfirmationNotFoundError,
)
from ccproxy.models.confirmations import ConfirmationEvent, ConfirmationStatus


logger = get_logger(__name__)


router = APIRouter(prefix="/api/v1/confirmations", tags=["confirmations"])


class ConfirmationResponse(BaseModel):
    """Response to a confirmation request."""

    allowed: bool


class ConfirmationRequestInfo(BaseModel):
    """Information about a confirmation request."""

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
    """Generate SSE events for confirmation requests.

    Args:
        request: The FastAPI request object

    Yields:
        Dict with event data for SSE
    """
    service = get_confirmation_service()
    queue = await service.subscribe_to_events()

    try:
        yield {
            "event": "ping",
            "data": json.dumps({"message": "Connected to confirmation stream"}),
        }

        # Send all pending confirmation requests to the newly connected client
        pending_requests = await service.get_pending_requests()
        for pending_req in pending_requests:
            event = ConfirmationEvent(
                type="confirmation_request",
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
                "event": "confirmation_request",
                "data": json.dumps(event.model_dump()),
            }

        while not await request.is_disconnected():
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=15.0)

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
async def stream_confirmations(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    auth: ConditionalAuthDep,
) -> EventSourceResponse:
    """Stream confirmation requests via Server-Sent Events.

    This endpoint streams new confirmation requests as they are created,
    allowing external tools to handle user confirmations.

    Returns:
        EventSourceResponse streaming confirmation events
    """
    return EventSourceResponse(
        event_generator(request),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/{confirmation_id}")
async def get_confirmation(
    confirmation_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
    auth: ConditionalAuthDep,
) -> ConfirmationRequestInfo:
    """Get information about a specific confirmation request.

    Args:
        confirmation_id: ID of the confirmation request

    Returns:
        Information about the confirmation request

    Raises:
        HTTPException: If request not found
    """
    service = get_confirmation_service()
    try:
        request = await service.get_request(confirmation_id)
        if not request:
            raise ConfirmationNotFoundError(confirmation_id)
    except ConfirmationNotFoundError as e:
        raise HTTPException(
            status_code=404, detail="Confirmation request not found"
        ) from e

    return ConfirmationRequestInfo(
        request_id=request.id,
        tool_name=request.tool_name,
        input=request.input,
        status=request.status.value,
        created_at=request.created_at.isoformat(),
        expires_at=request.expires_at.isoformat(),
        time_remaining=request.time_remaining(),
    )


@router.post("/{confirmation_id}/respond")
async def respond_to_confirmation(
    confirmation_id: str,
    response: ConfirmationResponse,
    settings: Annotated[Settings, Depends(get_settings)],
    auth: ConditionalAuthDep,
) -> dict[str, str | bool]:
    """Submit a response to a confirmation request.

    Args:
        confirmation_id: ID of the confirmation request
        response: The allow/deny response

    Returns:
        Success response

    Raises:
        HTTPException: If request not found or already resolved
    """
    service = get_confirmation_service()
    status = await service.get_status(confirmation_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Confirmation request not found")

    if status != ConfirmationStatus.PENDING:
        try:
            raise ConfirmationAlreadyResolvedError(confirmation_id, status.value)
        except ConfirmationAlreadyResolvedError as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=e.message,
            ) from e

    success = await service.resolve(confirmation_id, response.allowed)

    if not success:
        raise HTTPException(
            status_code=409, detail="Failed to resolve confirmation request"
        )

    logger.info(
        "confirmation_resolved_via_api",
        confirmation_id=confirmation_id,
        allowed=response.allowed,
    )

    return {
        "status": "success",
        "confirmation_id": confirmation_id,
        "allowed": response.allowed,
    }
