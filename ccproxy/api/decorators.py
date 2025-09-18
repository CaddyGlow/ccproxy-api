from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from fastapi import Request

from ccproxy.core.logging import get_logger as _get_logger
from ccproxy.core.request_context import RequestContext


P = ParamSpec("P")
R = TypeVar("R")


def format_chain(
    *formats: str,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Existing simple decorator to attach a format chain to a route handler.

    This attaches a __format_chain__ attribute used by validation and helpers.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        func.__format_chain__ = list(formats)  # type: ignore[attr-defined]

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_format_chain(
    formats: list[str], *, endpoint: str | None = None
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator to set format chain and optional endpoint metadata on a route.

    - Attaches __format_chain__ to the endpoint for upstream processing/validation
    - Ensures request.state.context exists and sets context.format_chain
    - Optionally sets context.metadata["endpoint"] to the provided upstream endpoint path
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        func.__format_chain__ = list(formats)  # type: ignore[attr-defined]

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Find Request in args/kwargs
            request: Request | None = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if request is None:
                req = kwargs.get("request")
                if isinstance(req, Request):
                    request = req

            if request is not None:
                # Ensure a context exists
                if (
                    not hasattr(request.state, "context")
                    or request.state.context is None
                ):
                    request.state.context = RequestContext(
                        request_id=str(uuid.uuid4()),
                        start_time=time.perf_counter(),
                        logger=_get_logger(__name__),
                    )
                # Set chain and endpoint metadata
                request.state.context.format_chain = list(formats)
                if endpoint:
                    request.state.context.metadata["endpoint"] = endpoint

            return await func(*args, **kwargs)

        return wrapper

    return decorator
