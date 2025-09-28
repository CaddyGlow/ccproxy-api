"""Context helpers for formatter conversions using async contextvars."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_REQUEST_VAR: ContextVar[Any | None] = ContextVar("formatter_request", default=None)
_INSTRUCTIONS_VAR: ContextVar[str | None] = ContextVar(
    "formatter_instructions", default=None
)


def register_request(request: Any | None, instructions: str | None = None) -> None:
    """Record the most recent upstream request for streaming conversions."""

    normalized = instructions.strip() if isinstance(instructions, str) else None

    _REQUEST_VAR.set(request)
    _INSTRUCTIONS_VAR.set(normalized)

    try:
        from ccproxy.core.request_context import RequestContext

        ctx = RequestContext.get_current()
        if ctx is not None:
            formatter_state = ctx.metadata.setdefault("formatter_state", {})
            if request is None:
                formatter_state.pop("request", None)
            else:
                formatter_state["request"] = request

            if normalized:
                formatter_state["instructions"] = normalized
            elif instructions is None:
                formatter_state.pop("instructions", None)
    except Exception:
        # Request context propagation is best-effort; proceed even when
        # request context is unavailable (e.g., during unit tests).
        pass


def get_last_request() -> Any | None:
    """Return the cached upstream request for the active conversion, if any."""

    try:
        from ccproxy.core.request_context import RequestContext

        ctx = RequestContext.get_current()
        if ctx is not None:
            formatter_state = ctx.metadata.get("formatter_state", {})
            if "request" in formatter_state:
                return formatter_state["request"]
    except Exception:
        pass

    return _REQUEST_VAR.get()


def get_last_instructions() -> str | None:
    """Return the cached instruction string from the last registered request."""

    try:
        from ccproxy.core.request_context import RequestContext

        ctx = RequestContext.get_current()
        if ctx is not None:
            formatter_state = ctx.metadata.get("formatter_state", {})
            instructions = formatter_state.get("instructions")
            if isinstance(instructions, str) and instructions.strip():
                return instructions.strip()
    except Exception:
        pass

    return _INSTRUCTIONS_VAR.get()
