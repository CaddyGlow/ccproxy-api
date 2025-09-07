"""Permissions plugin for CCProxy.

Provides permission management and authorization services.
"""

from .models import (
    EventType,
    PermissionEvent,
    PermissionRequest,
    PermissionStatus,
)
from .service import PermissionService, get_permission_service


__all__ = [
    "EventType",
    "PermissionEvent",
    "PermissionRequest",
    "PermissionService",
    "PermissionStatus",
    "get_permission_service",
]
