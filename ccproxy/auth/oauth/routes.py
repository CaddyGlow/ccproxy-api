"""OAuth authentication routes for Anthropic OAuth login."""

from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import ValidationError
from structlog import get_logger

from ccproxy.auth.exceptions import (
    CredentialsStorageError,
    OAuthError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.models import (
    ClaudeCredentials,
)
from ccproxy.auth.oauth.registry import get_oauth_registry


logger = get_logger(__name__)

router = APIRouter(tags=["oauth"])

# Store for pending OAuth flows
_pending_flows: dict[str, dict[str, Any]] = {}


def register_oauth_flow(
    state: str, code_verifier: str, custom_paths: list[Path] | None = None
) -> None:
    """Register a pending OAuth flow."""
    _pending_flows[state] = {
        "code_verifier": code_verifier,
        "custom_paths": custom_paths,
        "completed": False,
        "success": False,
        "error": None,
    }
    logger.debug(
        "Registered OAuth flow",
        state=state,
        operation="register_oauth_flow",
        category="auth",
    )


def get_oauth_flow_result(state: str) -> dict[str, Any] | None:
    """Get and remove OAuth flow result."""
    return _pending_flows.pop(state, None)


@router.get("/callback")
async def oauth_callback(
    request: Request,
    code: str | None = Query(None, description="Authorization code"),
    state: str | None = Query(None, description="State parameter"),
    error: str | None = Query(None, description="OAuth error"),
    error_description: str | None = Query(None, description="OAuth error description"),
) -> HTMLResponse:
    """Handle OAuth callback from Claude authentication.

    This endpoint receives the authorization code from Claude's OAuth flow
    and exchanges it for access tokens.
    """
    try:
        if error:
            error_msg = error_description or error or "OAuth authentication failed"
            logger.error(
                "OAuth callback error",
                error_type="oauth_error",
                error_message=error_msg,
                oauth_error=error,
                oauth_error_description=error_description,
                state=state,
                operation="oauth_callback",
                category="auth",
            )

            # Update pending flow if state is provided
            if state and state in _pending_flows:
                _pending_flows[state].update(
                    {
                        "completed": True,
                        "success": False,
                        "error": error_msg,
                    }
                )

            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Login Failed</title></head>
                    <body>
                        <h1>Login Failed</h1>
                        <p>Error: {error_msg}</p>
                        <p>You can close this window and try again.</p>
                    </body>
                </html>
                """,
                status_code=400,
            )

        if not code:
            error_msg = "No authorization code received"
            logger.error(
                "OAuth callback missing authorization code",
                error_type="missing_code",
                error_message=error_msg,
                state=state,
                operation="oauth_callback",
                category="auth",
            )

            if state and state in _pending_flows:
                _pending_flows[state].update(
                    {
                        "completed": True,
                        "success": False,
                        "error": error_msg,
                    }
                )

            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Login Failed</title></head>
                    <body>
                        <h1>Login Failed</h1>
                        <p>Error: {error_msg}</p>
                        <p>You can close this window and try again.</p>
                    </body>
                </html>
                """,
                status_code=400,
            )

        if not state:
            error_msg = "Missing state parameter"
            logger.error(
                "OAuth callback missing state parameter",
                error_type="missing_state",
                error_message=error_msg,
                operation="oauth_callback",
                category="auth",
            )
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Login Failed</title></head>
                    <body>
                        <h1>Login Failed</h1>
                        <p>Error: {error_msg}</p>
                        <p>You can close this window and try again.</p>
                    </body>
                </html>
                """,
                status_code=400,
            )

        # Check if this is a valid pending flow
        if state not in _pending_flows:
            error_msg = "Invalid or expired state parameter"
            logger.error(
                "OAuth callback with invalid state",
                error_type="invalid_state",
                error_message="Invalid or expired state parameter",
                state=state,
                operation="oauth_callback",
                category="auth",
            )
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Login Failed</title></head>
                    <body>
                        <h1>Login Failed</h1>
                        <p>Error: {error_msg}</p>
                        <p>You can close this window and try again.</p>
                    </body>
                </html>
                """,
                status_code=400,
            )

        # Get flow details
        flow = _pending_flows[state]
        code_verifier = flow["code_verifier"]
        custom_paths = flow["custom_paths"]

        # Exchange authorization code for tokens
        success = await _exchange_code_for_tokens(
            code, code_verifier, state, custom_paths
        )

        # Update flow result
        _pending_flows[state].update(
            {
                "completed": True,
                "success": success,
                "error": None if success else "Token exchange failed",
            }
        )

        if success:
            logger.info(
                "OAuth login successful",
                state=state,
                operation="oauth_callback",
                category="auth",
            )
            return HTMLResponse(
                content="""
                <html>
                    <head><title>Login Successful</title></head>
                    <body>
                        <h1>Login Successful!</h1>
                        <p>You have successfully logged in to Claude.</p>
                        <p>You can close this window and return to the CLI.</p>
                        <script>
                            setTimeout(() => {
                                window.close();
                            }, 3000);
                        </script>
                    </body>
                </html>
                """,
                status_code=200,
            )
        else:
            error_msg = "Failed to exchange authorization code for tokens"
            logger.error(
                "OAuth token exchange failed",
                error_type="token_exchange_failed",
                error_message=error_msg,
                state=state,
                operation="oauth_callback",
                category="auth",
            )
            return HTMLResponse(
                content=f"""
                <html>
                    <head><title>Login Failed</title></head>
                    <body>
                        <h1>Login Failed</h1>
                        <p>Error: {error_msg}</p>
                        <p>You can close this window and try again.</p>
                    </body>
                </html>
                """,
                status_code=500,
            )

    except (OAuthError, OAuthTokenRefreshError, CredentialsStorageError) as e:
        logger.error(
            "oauth_callback_error",
            error_type="auth_error",
            error=str(e),
            state=state,
            operation="oauth_callback",
            exc_info=e,
        )

        if state and state in _pending_flows:
            _pending_flows[state].update(
                {
                    "completed": True,
                    "success": False,
                    "error": str(e),
                }
            )

        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Login Error</title></head>
                <body>
                    <h1>Login Error</h1>
                    <p>Authentication error: {str(e)}</p>
                    <p>You can close this window and try again.</p>
                </body>
            </html>
            """,
            status_code=500,
        )
    except httpx.HTTPError as e:
        logger.error(
            "oauth_callback_http_error",
            error=str(e),
            status=e.response.status_code if hasattr(e, "response") else None,
            state=state,
            operation="oauth_callback",
            exc_info=e,
        )

        if state and state in _pending_flows:
            _pending_flows[state].update(
                {
                    "completed": True,
                    "success": False,
                    "error": f"HTTP error: {str(e)}",
                }
            )

        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Login Error</title></head>
                <body>
                    <h1>Login Error</h1>
                    <p>Network error occurred: {str(e)}</p>
                    <p>You can close this window and try again.</p>
                </body>
            </html>
            """,
            status_code=500,
        )
    except ValidationError as e:
        logger.error(
            "oauth_callback_validation_error",
            error=str(e),
            state=state,
            operation="oauth_callback",
            exc_info=e,
        )

        if state and state in _pending_flows:
            _pending_flows[state].update(
                {
                    "completed": True,
                    "success": False,
                    "error": f"Validation error: {str(e)}",
                }
            )

        return HTMLResponse(
            content="""
            <html>
                <head><title>Login Error</title></head>
                <body>
                    <h1>Login Error</h1>
                    <p>Data validation error occurred</p>
                    <p>You can close this window and try again.</p>
                </body>
            </html>
            """,
            status_code=500,
        )
    except Exception as e:
        logger.error(
            "oauth_callback_unexpected_error",
            error=str(e),
            state=state,
            operation="oauth_callback",
            exc_info=e,
        )

        if state and state in _pending_flows:
            _pending_flows[state].update(
                {
                    "completed": True,
                    "success": False,
                    "error": str(e),
                }
            )

        return HTMLResponse(
            content=f"""
            <html>
                <head><title>Login Error</title></head>
                <body>
                    <h1>Login Error</h1>
                    <p>An unexpected error occurred: {str(e)}</p>
                    <p>You can close this window and try again.</p>
                </body>
            </html>
            """,
            status_code=500,
        )


async def _exchange_code_for_tokens(
    authorization_code: str,
    code_verifier: str,
    state: str,
    custom_paths: list[Path] | None = None,
) -> bool:
    """Exchange authorization code for access tokens."""
    try:
        # Get OAuth provider from registry
        registry = get_oauth_registry()
        oauth_provider = registry.get_provider("claude-api")
        if not oauth_provider:
            logger.error("claude_oauth_provider_not_found", category="auth")
            return False

        # Use OAuth provider to handle the callback
        try:
            credentials = await oauth_provider.handle_callback(
                authorization_code, state, code_verifier
            )

            # Save credentials using plugin's token manager if custom paths specified
            if custom_paths:
                from ccproxy.auth.storage.generic import GenericJsonStorage
                from plugins.claude_api.auth.manager import ClaudeApiTokenManager

                # Use the first custom path for storage
                storage = GenericJsonStorage(custom_paths[0], ClaudeCredentials)
                manager = ClaudeApiTokenManager(storage=storage)
                if await manager.save_credentials(credentials):
                    logger.info(
                        "Successfully saved OAuth credentials to custom path",
                        operation="exchange_code_for_tokens",
                    )
                else:
                    logger.error(
                        "Failed to save OAuth credentials to custom path",
                        error_type="save_credentials_failed",
                        operation="exchange_code_for_tokens",
                    )

            logger.info(
                "OAuth flow completed successfully",
                operation="exchange_code_for_tokens",
            )
            return True

        except Exception as e:
            logger.error(
                "oauth_provider_callback_error",
                error=str(e),
                error_type=type(e).__name__,
                operation="exchange_code_for_tokens",
                exc_info=e,
            )
            return False

    except Exception as e:
        logger.error(
            "oauth_exchange_error",
            error=str(e),
            operation="exchange_code_for_tokens",
            exc_info=e,
        )
        return False
