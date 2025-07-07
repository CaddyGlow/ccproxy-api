"""OAuth authentication routes for Claude login."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from claude_code_proxy.services.credentials import CredentialsService
from claude_code_proxy.utils.logging import get_logger


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
    logger.debug(f"Registered OAuth flow for state: {state}")


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
            logger.error(f"OAuth callback error: {error_msg}")

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
            logger.error(error_msg)

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
            logger.error(error_msg)
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
            logger.error(f"OAuth callback with unknown state: {state}")
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
            logger.info("OAuth login successful")
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
            logger.error(error_msg)
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
        logger.exception("Unexpected error in OAuth callback")

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
        from pathlib import Path

        import httpx

        from claude_code_proxy.services.credentials import ClaudeCredentials, OAuthToken

        # Exchange authorization code for tokens
        token_data = {
            "grant_type": "authorization_code",
            "code": authorization_code,
            "redirect_uri": CredentialsService.REDIRECT_URI,
            "client_id": CredentialsService.CLIENT_ID,
            "code_verifier": code_verifier,
        }

        headers = {
            "Content-Type": "application/json",
            "anthropic-beta": CredentialsService.OAUTH_BETA_VERSION,
            "User-Agent": "Claude-Code/1.0.43",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                CredentialsService.TOKEN_URL,
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
                    else CredentialsService.SCOPES,
                    "subscriptionType": result.get("subscription_type", "unknown"),
                }

                credentials = ClaudeCredentials(claudeAiOauth=OAuthToken(**oauth_data))

                # Save credentials
                if custom_paths:
                    # For custom paths, ensure directory exists
                    for path in custom_paths:
                        Path(path).parent.mkdir(parents=True, exist_ok=True)

                if CredentialsService.save_credentials(credentials, custom_paths):
                    logger.info("Successfully saved OAuth credentials")
                    return True
                else:
                    logger.error("Failed to save OAuth credentials")
                    return False

            else:
                logger.error(
                    f"Token exchange failed: {response.status_code} - {response.text}"
                )
                return False

    except Exception as e:
        logger.exception("Error during token exchange")
        return False
