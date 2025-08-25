"""Event definitions for the hook system."""

from enum import Enum


class HookEvent(str, Enum):
    """Event types that can trigger hooks"""

    # Application Lifecycle
    APP_STARTUP = "app.startup"
    APP_SHUTDOWN = "app.shutdown"
    APP_READY = "app.ready"

    # Request Lifecycle
    REQUEST_STARTED = "request.started"
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"

    # Provider Integration
    PROVIDER_REQUEST_SENT = "provider.request.sent"
    PROVIDER_RESPONSE_RECEIVED = "provider.response.received"
    PROVIDER_ERROR = "provider.error"
    PROVIDER_STREAM_START = "provider.stream.start"
    PROVIDER_STREAM_CHUNK = "provider.stream.chunk"
    PROVIDER_STREAM_END = "provider.stream.end"

    # Plugin Management
    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_UNLOADED = "plugin.unloaded"
    PLUGIN_ERROR = "plugin.error"

    # Custom Events
    CUSTOM_EVENT = "custom.event"
