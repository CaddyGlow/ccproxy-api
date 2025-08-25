"""Central OAuth router that delegates to plugin providers.

This module provides unified OAuth endpoints that dynamically route
to the appropriate plugin-based OAuth provider.
"""

import secrets

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from ccproxy.auth.oauth.registry import OAuthProviderInfo, get_oauth_registry
from ccproxy.auth.oauth.session import get_oauth_session_manager
from ccproxy.auth.oauth.templates import OAuthTemplates


logger = structlog.get_logger(__name__)

# Create the OAuth router
oauth_router = APIRouter(prefix="/oauth", tags=["oauth"])


class OAuthProvidersResponse(BaseModel):
    """Response for listing OAuth providers."""

    providers: dict[str, OAuthProviderInfo]


class OAuthLoginResponse(BaseModel):
    """Response for OAuth login initiation."""

    auth_url: str
    state: str
    provider: str


class OAuthErrorResponse(BaseModel):
    """Response for OAuth errors."""

    error: str
    error_description: str | None = None
    provider: str | None = None


@oauth_router.get("/providers", response_model=OAuthProvidersResponse)
async def list_oauth_providers() -> OAuthProvidersResponse:
    """List all available OAuth providers.

    Returns:
        Dictionary of available OAuth providers with their information
    """
    registry = get_oauth_registry()
    providers = registry.list_providers()

    logger.info("oauth_providers_listed", count=len(providers), category="auth")

    return OAuthProvidersResponse(providers=providers)


@oauth_router.get("/{provider}/login")
async def initiate_oauth_login(
    provider: str,
    redirect_uri: str | None = Query(
        None, description="Optional redirect URI override"
    ),
    scopes: str | None = Query(
        None, description="Optional scope override (comma-separated)"
    ),
) -> RedirectResponse:
    """Initiate OAuth login flow for a specific provider.

    Args:
        provider: Provider name (e.g., 'claude-api', 'codex')
        redirect_uri: Optional redirect URI override
        scopes: Optional scope override

    Returns:
        Redirect to provider's authorization URL

    Raises:
        HTTPException: If provider not found or error generating auth URL
    """
    registry = get_oauth_registry()
    oauth_provider = registry.get_provider(provider)

    if not oauth_provider:
        logger.error("oauth_provider_not_found", provider=provider, category="auth")
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider '{provider}' not found",
        )

    # Generate OAuth state for CSRF protection
    state = secrets.token_urlsafe(32)

    # Generate PKCE code verifier if provider supports it
    code_verifier = None
    if oauth_provider.supports_pkce:
        # Generate PKCE pair
        import base64

        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )

    # Store OAuth session data
    session_manager = get_oauth_session_manager()
    session_data = {
        "provider": provider,
        "state": state,
        "redirect_uri": redirect_uri,
        "scopes": scopes.split(",") if scopes else None,
    }
    if code_verifier:
        session_data["code_verifier"] = code_verifier

    await session_manager.create_session(state, session_data)

    try:
        # Get authorization URL from provider
        auth_url = await oauth_provider.get_authorization_url(state, code_verifier)

        logger.info(
            "oauth_login_initiated",
            provider=provider,
            state=state,
            has_pkce=bool(code_verifier),
            category="auth",
        )

        # Redirect to provider's authorization page
        return RedirectResponse(url=auth_url, status_code=302)

    except Exception as e:
        logger.error(
            "oauth_login_error",
            provider=provider,
            error=str(e),
            exc_info=e,
            category="auth",
        )
        await session_manager.delete_session(state)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate OAuth login: {str(e)}",
        ) from e


@oauth_router.get("/{provider}/callback")
async def handle_oauth_callback(
    provider: str,
    request: Request,
    code: str | None = Query(None, description="Authorization code"),
    state: str | None = Query(None, description="OAuth state"),
    error: str | None = Query(None, description="OAuth error"),
    error_description: str | None = Query(None, description="Error description"),
) -> HTMLResponse:
    """Handle OAuth callback from provider.

    Args:
        provider: Provider name
        request: FastAPI request
        code: Authorization code from provider
        state: OAuth state for validation
        error: OAuth error code
        error_description: OAuth error description

    Returns:
        HTML response with success or error message

    Raises:
        HTTPException: If provider not found or callback handling fails
    """
    # Handle OAuth errors
    if error:
        logger.error(
            "oauth_callback_error",
            provider=provider,
            error=error,
            error_description=error_description,
            category="auth",
        )

        return OAuthTemplates.callback_error(
            error=error,
            error_description=error_description,
        )

    # Validate required parameters
    if not code or not state:
        logger.error(
            "oauth_callback_missing_params",
            provider=provider,
            has_code=bool(code),
            has_state=bool(state),
            category="auth",
        )
        return OAuthTemplates.error(
            error_message="No authorization code was received.",
            title="Missing Authorization Code",
            error_detail="The OAuth server did not provide an authorization code. Please try again.",
            status_code=400,
        )

    # Get OAuth session
    session_manager = get_oauth_session_manager()
    session_data = await session_manager.get_session(state)

    if not session_data:
        logger.error(
            "oauth_callback_invalid_state",
            provider=provider,
            state=state,
            category="auth",
        )
        return OAuthTemplates.error(
            error_message="The authentication state is invalid or has expired.",
            title="Invalid State",
            error_detail="This may indicate a CSRF attack or an expired authentication session. Please start the authentication process again.",
            status_code=400,
        )

    # Validate provider matches
    if session_data.get("provider") != provider:
        logger.error(
            "oauth_callback_provider_mismatch",
            expected=session_data.get("provider"),
            actual=provider,
            category="auth",
        )
        await session_manager.delete_session(state)
        return OAuthTemplates.error(
            error_message="Provider mismatch in OAuth callback",
        )

    # Get provider instance
    registry = get_oauth_registry()
    oauth_provider = registry.get_provider(provider)

    if not oauth_provider:
        logger.error("oauth_provider_not_found", provider=provider, category="auth")
        await session_manager.delete_session(state)
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider '{provider}' not found",
        )

    try:
        # Exchange code for tokens
        code_verifier = session_data.get("code_verifier")
        credentials = await oauth_provider.handle_callback(code, state, code_verifier)

        # Clean up session
        await session_manager.delete_session(state)

        logger.info(
            "oauth_callback_success",
            provider=provider,
            has_credentials=bool(credentials),
            category="auth",
        )

        # Return success page
        return OAuthTemplates.success(
            message="Authentication successful! You can close this window.",
        )

    except Exception as e:
        logger.error(
            "oauth_callback_exchange_error",
            provider=provider,
            error=str(e),
            exc_info=e,
            category="auth",
        )
        await session_manager.delete_session(state)

        return OAuthTemplates.error(
            error_message="Failed to exchange authorization code for tokens.",
            title="Token Exchange Failed",
            error_detail=str(e),
            status_code=500,
        )


@oauth_router.post("/{provider}/refresh")
async def refresh_oauth_token(
    provider: str,
    refresh_token: str,
) -> JSONResponse:
    """Refresh OAuth access token.

    Args:
        provider: Provider name
        refresh_token: Refresh token

    Returns:
        New token response

    Raises:
        HTTPException: If provider not found or refresh fails
    """
    registry = get_oauth_registry()
    oauth_provider = registry.get_provider(provider)

    if not oauth_provider:
        logger.error("oauth_provider_not_found", provider=provider, category="auth")
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider '{provider}' not found",
        )

    try:
        new_tokens = await oauth_provider.refresh_access_token(refresh_token)

        logger.info("oauth_token_refreshed", provider=provider, category="auth")

        return JSONResponse(content=new_tokens, status_code=200)

    except Exception as e:
        logger.error(
            "oauth_refresh_error",
            provider=provider,
            error=str(e),
            exc_info=e,
            category="auth",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh token: {str(e)}",
        ) from e


@oauth_router.post("/{provider}/revoke")
async def revoke_oauth_token(
    provider: str,
    token: str,
) -> Response:
    """Revoke an OAuth token.

    Args:
        provider: Provider name
        token: Token to revoke

    Returns:
        Empty response on success

    Raises:
        HTTPException: If provider not found or revocation fails
    """
    registry = get_oauth_registry()
    oauth_provider = registry.get_provider(provider)

    if not oauth_provider:
        logger.error("oauth_provider_not_found", provider=provider, category="auth")
        raise HTTPException(
            status_code=404,
            detail=f"OAuth provider '{provider}' not found",
        )

    try:
        await oauth_provider.revoke_token(token)

        logger.info("oauth_token_revoked", provider=provider, category="auth")

        return Response(status_code=204)

    except Exception as e:
        logger.error(
            "oauth_revoke_error",
            provider=provider,
            error=str(e),
            exc_info=e,
            category="auth",
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to revoke token: {str(e)}",
        ) from e
