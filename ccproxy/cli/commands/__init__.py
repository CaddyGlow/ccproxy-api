"""Command modules for CCProxy API CLI."""

from .auth import app as auth_app
from .config import app as config_app
from .serve import api


__all__ = [
    "api",
    "auth_app",
    "config_app",
]
