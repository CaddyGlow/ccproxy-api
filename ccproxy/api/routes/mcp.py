"""MCP (Model Context Protocol) server for CCProxy API Server.

Provides MCP server functionality including permission checking tools.
"""

import asyncio
import json
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from fastapi_mcp import FastApiMCP  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field
from structlog import get_logger

from ccproxy.api.services.confirmation_service import (
    ConfirmationStatus,
    get_confirmation_service,
)
from ccproxy.config.settings import Settings, get_settings
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
    input: Annotated[dict[str, Any], Field(description="Input parameters for the tool")]
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
    settings: Annotated[Settings, Depends(get_settings)],
) -> (
    PermissionToolAllowResponse
    | PermissionToolDenyResponse
    | PermissionToolPendingResponse
):
    """Check permissions for a tool call.

    This implements the same security logic as the CLI permission tool,
    checking for dangerous patterns and restricted tools.

    Args:
        request: The permission check request containing tool name and input
        settings: Application settings

    Returns:
        Permission response indicating allow/deny with appropriate details
    """
    # Log the permission check request
    logger.info(
        "permission_check_request_received",
        tool_name=request.tool_name,
        input_keys=list(request.input.keys()),
        has_confirmation_id=request.confirmation_id is not None,
    )

    confirmation_service = get_confirmation_service()

    # Check if this is a retry with confirmation ID
    if request.confirmation_id:
        status = confirmation_service.get_status(request.confirmation_id)

        if status == ConfirmationStatus.ALLOWED:
            logger.info(
                "permission_allowed_after_confirmation",
                tool_name=request.tool_name,
                confirmation_id=request.confirmation_id,
            )
            return PermissionToolAllowResponse(updated_input=request.input)

        elif status == ConfirmationStatus.DENIED:
            logger.info(
                "permission_denied_after_confirmation",
                tool_name=request.tool_name,
                confirmation_id=request.confirmation_id,
            )
            return PermissionToolDenyResponse(message="User denied the operation")

        elif status == ConfirmationStatus.EXPIRED:
            logger.info(
                "permission_expired",
                tool_name=request.tool_name,
                confirmation_id=request.confirmation_id,
            )
            return PermissionToolDenyResponse(message="Confirmation request expired")

        # If pending or not found, continue with normal flow

    # Always request user confirmation for ALL commands
    logger.info(
        "permission_requires_confirmation",
        tool_name=request.tool_name,
    )

    # Request user confirmation for all operations
    confirmation_id = await confirmation_service.request_confirmation(
        tool_name=request.tool_name,
        input=request.input,
    )

    # Wait for the confirmation to be resolved instead of returning pending
    try:
        logger.debug(
            "waiting_for_confirmation_resolution",
            confirmation_id=confirmation_id,
            tool_name=request.tool_name,
        )

        final_status = await confirmation_service.wait_for_confirmation(confirmation_id)

        if final_status == ConfirmationStatus.ALLOWED:
            logger.info(
                "permission_allowed_after_wait",
                tool_name=request.tool_name,
                confirmation_id=confirmation_id,
            )
            return PermissionToolAllowResponse(updated_input=request.input)

        elif final_status == ConfirmationStatus.DENIED:
            logger.info(
                "permission_denied_after_wait",
                tool_name=request.tool_name,
                confirmation_id=confirmation_id,
            )
            return PermissionToolDenyResponse(message="User denied the operation")

        else:  # EXPIRED
            logger.info(
                "permission_expired_after_wait",
                tool_name=request.tool_name,
                confirmation_id=confirmation_id,
            )
            return PermissionToolDenyResponse(message="Confirmation request expired")

    except TimeoutError:
        logger.warning(
            "permission_wait_timeout",
            tool_name=request.tool_name,
            confirmation_id=confirmation_id,
        )
        return PermissionToolDenyResponse(message="Confirmation request timed out")

    except Exception as e:
        logger.error(
            "permission_wait_error",
            tool_name=request.tool_name,
            confirmation_id=confirmation_id,
            error=str(e),
            exc_info=True,
        )
        return PermissionToolDenyResponse(message="Error waiting for confirmation")


def setup_mcp(app: FastAPI) -> None:
    """Set up MCP server on the given FastAPI app.

    Args:
        app: The FastAPI application to mount MCP on
    """
    # Create a clean sub-application for MCP without any middleware
    mcp_app = FastAPI(
        title="CCProxy MCP Server",
        description="MCP server for Claude Code permission checking",
        # Disable automatic validation exception handlers to avoid middleware
        openapi_url=None,  # Disable OpenAPI to keep it minimal
        docs_url=None,
        redoc_url=None,
    )

    # Add the permission checking endpoint to the MCP app
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
        settings: Annotated[Settings, Depends(get_settings)],
    ) -> (
        PermissionToolAllowResponse
        | PermissionToolDenyResponse
        | PermissionToolPendingResponse
    ):
        """Check permissions for a tool call."""
        return await check_permission(request, settings)

    # Create MCP server instance from the MCP app
    mcp = FastApiMCP(
        mcp_app,
        name="CCProxy MCP Server",
        description="MCP server for Claude Code permission checking",
        # Only expose the permission checking endpoint
        include_operations=["check_permission"],
    )

    # Mount the MCP server to the main app
    mcp.mount(app, mount_path="/mcp")

    logger.info("mcp_app_mounted", mount_path="/mcp")
