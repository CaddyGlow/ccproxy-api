"""OAuth authentication routes for Anthropic OAuth login."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from structlog import get_logger

from ccproxy.auth.models import (
    ClaudeCredentials,
    OAuthToken,
)
from ccproxy.auth.storage import JsonFileTokenStorage as JsonFileStorage

# Import CredentialsManager locally to avoid circular import
from ccproxy.services.credentials.config import OAuthConfig


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
    logger.debug("Registered OAuth flow", state=state, operation="register_oauth_flow")


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
        success = await _exchange_code_for_tokens(code, code_verifier, custom_paths)

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
                "OAuth login successful", state=state, operation="oauth_callback"
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

    except Exception as e:
        logger.error(
            "Unexpected error in OAuth callback",
            error_type="unexpected_error",
            error_message=str(e),
            state=state,
            operation="oauth_callback",
            exc_info=True,
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
    authorization_code: str, code_verifier: str, custom_paths: list[Path] | None = None
) -> bool:
    """Exchange authorization code for access tokens."""
    try:
        from datetime import UTC, datetime

        import httpx

        # Create OAuth config with default values
        oauth_config = OAuthConfig()

        # Exchange authorization code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": oauth_config.redirect_uri,
            "client_id": oauth_config.client_id,
            "code_verifier": code_verifier,
        }

        headers = {
            "Content-Type": "application/json",
            "anthropic-beta": oauth_config.beta_version,
            "User-Agent": oauth_config.user_agent,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                oauth_config.token_url,
                headers=headers,
                json=token_data,
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()

                # Calculate expires_at from expires_in
                expires_in = result.get("expires_in")
                expires_at = None
                if expires_in:
                    expires_at = int(
                        (datetime.now(UTC).timestamp() + expires_in) * 1000
                    )

                # Create credentials object
                oauth_data = {
                    "accessToken": result.get("access_token"),
                    "refreshToken": result.get("refresh_token"),
                    "expiresAt": expires_at,
                    "scopes": result.get("scope", "").split()
                    if result.get("scope")
                    else oauth_config.scopes,
                    "subscriptionType": result.get("subscription_type", "unknown"),
                }

                credentials = ClaudeCredentials(claudeAiOauth=OAuthToken(**oauth_data))

                # Save credentials using CredentialsManager (lazy import to avoid circular import)
                from ccproxy.services.credentials.manager import CredentialsManager

                if custom_paths:
                    # Use the first custom path for storage
                    storage = JsonFileStorage(custom_paths[0])
                    manager = CredentialsManager(storage=storage)
                else:
                    manager = CredentialsManager()

                if await manager.save(credentials):
                    logger.info(
                        "Successfully saved OAuth credentials",
                        subscription_type=oauth_data["subscriptionType"],
                        scopes=oauth_data["scopes"],
                        operation="exchange_code_for_tokens",
                    )
                    return True
                else:
                    logger.error(
                        "Failed to save OAuth credentials",
                        error_type="save_credentials_failed",
                        operation="exchange_code_for_tokens",
                    )
                    return False

            else:
                # Use compact logging for the error message
                import os

                verbose_api = (
                    os.environ.get("CCPROXY_VERBOSE_API", "false").lower() == "true"
                )

                if verbose_api:
                    error_detail = response.text
                else:
                    response_text = response.text
                    if len(response_text) > 200:
                        error_detail = f"{response_text[:100]}...{response_text[-50:]}"
                    elif len(response_text) > 100:
                        error_detail = f"{response_text[:100]}..."
                    else:
                        error_detail = response_text

                logger.error(
                    "Token exchange failed",
                    error_type="token_exchange_failed",
                    status_code=response.status_code,
                    error_detail=error_detail,
                    verbose_api_enabled=verbose_api,
                    operation="exchange_code_for_tokens",
                )
                return False

    except Exception as e:
        logger.error(
            "Error during token exchange",
            error_type="token_exchange_exception",
            error_message=str(e),
            operation="exchange_code_for_tokens",
            exc_info=True,
        )
        return False
