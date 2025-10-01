"""FastAPI dependency injection utilities for authentication."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ccproxy.auth.bearer import BearerTokenAuthManager
from ccproxy.auth.exceptions import AuthenticationError, AuthenticationRequiredError
from ccproxy.auth.manager import AuthManager
from ccproxy.config.settings import Settings


if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    pass


# FastAPI security scheme for bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


def get_settings() -> Settings:
    """Get settings instance directly (without service container)."""
    return Settings()


SettingsDep = Annotated[Settings, Depends(get_settings)]


def _resolve_runtime_settings(request: Request) -> Settings | None:
    """Best-effort retrieval of settings without importing API dependencies."""

    container = getattr(request.app.state, "service_container", None)
    if container is None:
        try:
            from ccproxy.services.container import ServiceContainer

            container = ServiceContainer.get_current(strict=False)
        except (ImportError, RuntimeError):
            container = None

    if container is not None:
        try:
            return container.get_service(Settings)
        except ValueError:
            # Service not registered yet; fall through to default settings.
            pass

    try:
        return Settings()
    except Exception:
        # Settings construction can fail in minimal test contexts.
        return None


def _expected_token(settings: Settings | None) -> str | None:
    """Extract configured auth token from settings if present."""

    if settings and settings.security.auth_token:
        return settings.security.auth_token.get_secret_value()
    return None


async def _build_bearer_auth_manager(
    credentials: HTTPAuthorizationCredentials | None,
    expected_token: str | None,
    *,
    require_credentials: bool,
) -> AuthManager | None:
    """Create a bearer auth manager when credentials satisfy expectations."""

    token = credentials.credentials if credentials and credentials.credentials else None

    if token is None:
        if require_credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return None

    if expected_token is not None and token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        bearer_auth = BearerTokenAuthManager(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    try:
        if await bearer_auth.is_authenticated():
            return bearer_auth
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    if require_credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


async def get_auth_manager(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    settings: SettingsDep,
) -> AuthManager:
    """Return an authentication manager when credentials are required."""

    auth_manager = await _build_bearer_auth_manager(
        credentials,
        _expected_token(settings),
        require_credentials=True,
    )
    # require_credentials ensures auth_manager is never None here.
    assert auth_manager is not None
    return auth_manager


async def require_auth(
    auth_manager: Annotated[AuthManager, Depends(get_auth_manager)],
) -> AuthManager:
    """Enforce authentication, translating failures into HTTP errors."""

    try:
        if not await auth_manager.is_authenticated():
            raise AuthenticationRequiredError("Authentication required")
        return auth_manager
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_access_token(
    auth_manager: Annotated[AuthManager, Depends(require_auth)],
) -> str:
    """Retrieve an access token from an authenticated manager."""

    try:
        return await auth_manager.get_access_token()
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_conditional_auth_manager(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthManager | None:
    """Return an auth manager only when the configuration requires it."""

    settings = _resolve_runtime_settings(request)
    expected_token = _expected_token(settings)

    if expected_token is None:
        return None

    return await _build_bearer_auth_manager(
        credentials,
        expected_token,
        require_credentials=True,
    )


# Type aliases for common dependencies
AuthManagerDep = Annotated[AuthManager, Depends(get_auth_manager)]
RequiredAuthDep = Annotated[AuthManager, Depends(require_auth)]
AccessTokenDep = Annotated[str, Depends(get_access_token)]
ConditionalAuthDep = Annotated[
    AuthManager | None, Depends(get_conditional_auth_manager)
]
