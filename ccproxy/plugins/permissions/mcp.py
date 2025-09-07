"""MCP (Model Context Protocol) server for CCProxy API Server.

Provides MCP server functionality including permission checking tools.
"""

from typing import Annotated

from fastapi import APIRouter, FastAPI
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, ConfigDict, Field

from ccproxy.api.dependencies import SettingsDep
from ccproxy.core.logging import get_plugin_logger

from .models import (
    PermissionStatus,
    PermissionToolAllowResponse,
    PermissionToolDenyResponse,
    PermissionToolPendingResponse,
)
from .service import get_permission_service


logger = get_plugin_logger()


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
    permission_id: Annotated[
        str | None,
        Field(
            description="ID of a previous permission request for retry",
            alias="permissionId",
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
        retry=request.permission_id is not None,
    )

    permission_service = get_permission_service()

    if request.permission_id:
        status = await permission_service.get_status(request.permission_id)

        if status == PermissionStatus.ALLOWED:
            return PermissionToolAllowResponse(updated_input=request.input)

        elif status == PermissionStatus.DENIED:
            return PermissionToolDenyResponse(message="User denied the operation")

        elif status == PermissionStatus.EXPIRED:
            return PermissionToolDenyResponse(message="Permission request expired")

    logger.info(
        "permission_requires_authorization",
        tool_name=request.tool_name,
    )

    permission_id = await permission_service.request_permission(
        tool_name=request.tool_name,
        input=request.input,
    )

    # Wait for permission to be resolved
    try:
        final_status = await permission_service.wait_for_permission(
            permission_id,
            timeout_seconds=settings.security.confirmation_timeout_seconds,
        )

        if final_status == PermissionStatus.ALLOWED:
            logger.info(
                "permission_allowed_after_authorization",
                tool_name=request.tool_name,
                permission_id=permission_id,
            )
            return PermissionToolAllowResponse(updated_input=request.input)
        else:
            logger.info(
                "permission_denied_after_authorization",
                tool_name=request.tool_name,
                permission_id=permission_id,
                status=final_status.value,
            )
            return PermissionToolDenyResponse(
                message=f"User denied the operation (status: {final_status.value})"
            )

    except TimeoutError:
        logger.warning(
            "permission_authorization_timeout",
            tool_name=request.tool_name,
            permission_id=permission_id,
            timeout_seconds=settings.security.confirmation_timeout_seconds,
        )
        return PermissionToolDenyResponse(message="Permission request timed out")


# Create a router for the plugin system

mcp_router = APIRouter()


@mcp_router.post(
    "/permission/check",
    operation_id="check_permission",
    summary="Check permissions for a tool call",
    description="Validates whether a tool call should be allowed based on security rules",
    response_model=PermissionToolAllowResponse
    | PermissionToolDenyResponse
    | PermissionToolPendingResponse,
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
