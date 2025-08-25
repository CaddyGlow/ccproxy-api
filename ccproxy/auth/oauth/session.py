"""OAuth session management for handling OAuth state and PKCE.

This module provides session management for OAuth flows, storing
state, PKCE verifiers, and other session data during the OAuth process.
"""

import time
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


class OAuthSessionManager:
    """Manages OAuth session data during authentication flows.

    This is a simple in-memory implementation. In production,
    consider using Redis or another persistent store.
    """

    def __init__(self, ttl_seconds: int = 600) -> None:
        """Initialize the session manager.

        Args:
            ttl_seconds: Time-to-live for sessions in seconds (default: 10 minutes)
        """
        self._sessions: dict[str, dict[str, Any]] = {}
        self._ttl_seconds = ttl_seconds
        logger.info(
            "oauth_session_manager_initialized",
            ttl_seconds=ttl_seconds,
            category="auth",
        )

    async def create_session(self, state: str, data: dict[str, Any]) -> None:
        """Create a new OAuth session.

        Args:
            state: OAuth state parameter (session key)
            data: Session data to store
        """
        self._sessions[state] = {
            **data,
            "created_at": time.time(),
        }
        logger.debug(
            "oauth_session_created",
            state=state,
            provider=data.get("provider"),
            has_pkce=bool(data.get("code_verifier")),
            category="auth",
        )

        # Clean up expired sessions
        await self._cleanup_expired()

    async def get_session(self, state: str) -> dict[str, Any] | None:
        """Retrieve session data by state.

        Args:
            state: OAuth state parameter

        Returns:
            Session data or None if not found/expired
        """
        session = self._sessions.get(state)

        if not session:
            logger.debug("oauth_session_not_found", state=state, category="auth")
            return None

        # Check if session expired
        created_at = session.get("created_at", 0)
        if time.time() - created_at > self._ttl_seconds:
            logger.debug("oauth_session_expired", state=state, category="auth")
            await self.delete_session(state)
            return None

        logger.debug(
            "oauth_session_retrieved",
            state=state,
            provider=session.get("provider"),
            category="auth",
        )
        return session

    async def delete_session(self, state: str) -> None:
        """Delete a session.

        Args:
            state: OAuth state parameter
        """
        if state in self._sessions:
            provider = self._sessions[state].get("provider")
            del self._sessions[state]
            logger.debug(
                "oauth_session_deleted", state=state, provider=provider, category="auth"
            )

    async def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        current_time = time.time()
        expired_states = [
            state
            for state, session in self._sessions.items()
            if current_time - session.get("created_at", 0) > self._ttl_seconds
        ]

        for state in expired_states:
            await self.delete_session(state)

        if expired_states:
            logger.debug(
                "oauth_sessions_cleaned", count=len(expired_states), category="auth"
            )

    def clear_all(self) -> None:
        """Clear all sessions (mainly for testing)."""
        count = len(self._sessions)
        self._sessions.clear()
        logger.info("oauth_sessions_cleared", count=count, category="auth")


# Global session manager instance
_session_manager: OAuthSessionManager | None = None


def get_oauth_session_manager() -> OAuthSessionManager:
    """Get the global OAuth session manager instance.

    Returns:
        Global OAuth session manager
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = OAuthSessionManager()
    return _session_manager


def reset_oauth_session_manager() -> None:
    """Reset the global OAuth session manager.

    This clears all sessions and creates a new manager.
    Mainly useful for testing.
    """
    global _session_manager
    if _session_manager:
        _session_manager.clear_all()
    _session_manager = OAuthSessionManager()
