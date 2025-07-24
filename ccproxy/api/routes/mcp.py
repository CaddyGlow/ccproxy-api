"""MCP (Model Context Protocol) server for CCProxy API Server.

Provides MCP server functionality including permission checking tools.
"""

import asyncio
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi_mcp import FastApiMCP  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from structlog import get_logger

from ccproxy.api.dependencies import SettingsDep
from ccproxy.api.services.confirmation_service import get_confirmation_service
from ccproxy.models.confirmations import ConfirmationStatus
from ccproxy.models.responses import (
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
    PermissionToolPendingResponse,
)


logger = get_logger(__name__)


class PermissionCheckRequest(BaseModel):
    """Request model for permission checking."""

    tool_name: Annotated[
        str, Field(description="Name of the tool to check permissions for")
    ]
    input: Annotated[dict[str, str], Field(description="Input parameters for the tool")]
    tool_use_id: Annotated[
        str | None,
        Field(
            description="Id of the tool execution",
        ),
    ] = None
    confirmation_id: Annotated[
        str | None,
        Field(
            description="ID of a previous confirmation request for retry",
            alias="confirmationId",
        ),
    ] = None

    model_config = ConfigDict(populate_by_name=True)


async def check_permission(
    request: PermissionCheckRequest,
    settings: SettingsDep,
) -> (
    PermissionToolAllowResponse
    | PermissionToolDenyResponse
    | PermissionToolPendingResponse
):
    """Check permissions for a tool call.

    This implements the same security logic as the CLI permission tool,
    checking for dangerous patterns and restricted tools.
    """
    logger.info(
        "permission_check",
        tool_name=request.tool_name,
        retry=request.confirmation_id is not None,
    )

    confirmation_service = get_confirmation_service()

    if request.confirmation_id:
        status = await confirmation_service.get_status(request.confirmation_id)

        if status == ConfirmationStatus.ALLOWED:
            return PermissionToolAllowResponse(updated_input=request.input)

        elif status == ConfirmationStatus.DENIED:
            return PermissionToolDenyResponse(message="User denied the operation")

        elif status == ConfirmationStatus.EXPIRED:
            return PermissionToolDenyResponse(message="Confirmation request expired")

    logger.info(
        "permission_requires_confirmation",
        tool_name=request.tool_name,
    )

    confirmation_id = await confirmation_service.request_confirmation(
        tool_name=request.tool_name,
        input=request.input,
    )

    # Wait for confirmation to be resolved
    try:
        final_status = await confirmation_service.wait_for_confirmation(
            confirmation_id,
            timeout_seconds=settings.security.confirmation_timeout_seconds,
        )

        if final_status == ConfirmationStatus.ALLOWED:
            logger.info(
                "permission_allowed_after_confirmation",
                tool_name=request.tool_name,
                confirmation_id=confirmation_id,
            )
            return PermissionToolAllowResponse(updated_input=request.input)
        else:
            logger.info(
                "permission_denied_after_confirmation",
                tool_name=request.tool_name,
                confirmation_id=confirmation_id,
                status=final_status.value,
            )
            return PermissionToolDenyResponse(
                message=f"User denied the operation (status: {final_status.value})"
            )

    except TimeoutError:
        logger.warning(
            "permission_confirmation_timeout",
            tool_name=request.tool_name,
            confirmation_id=confirmation_id,
            timeout_seconds=settings.security.confirmation_timeout_seconds,
        )
        return PermissionToolDenyResponse(message="Confirmation request timed out")


def setup_mcp(app: FastAPI) -> None:
    """Set up MCP server on the given FastAPI app.

    Args:
        app: The FastAPI application to mount MCP on
    """
    # Minimal MCP sub-app without middleware or docs
    mcp_app = FastAPI(
        title="CCProxy MCP Server",
        description="MCP server for Claude Code permission checking",
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
    )

    @mcp_app.post(
        "/permission/check",
        operation_id="check_permission",
        summary="Check permissions for a tool call",
        description="Validates whether a tool call should be allowed based on security rules",
        response_model=PermissionToolAllowResponse
        | PermissionToolDenyResponse
        | PermissionToolPendingResponse,
        tags=["mcp-tools"],
    )
    async def permission_endpoint(
        request: PermissionCheckRequest,
        settings: SettingsDep,
    ) -> (
        PermissionToolAllowResponse
        | PermissionToolDenyResponse
        | PermissionToolPendingResponse
    ):
        """Check permissions for a tool call."""
        return await check_permission(request, settings)

    mcp = FastApiMCP(
        mcp_app,
        name="CCProxy MCP Server",
        description="MCP server for Claude Code permission checking",
        include_operations=["check_permission"],
    )

    mcp.mount(app, mount_path="/mcp")
