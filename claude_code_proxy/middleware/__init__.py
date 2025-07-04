"""Authentication middleware for Claude Proxy API."""

from .auth import get_auth_dependency


__all__ = ["get_auth_dependency"]
