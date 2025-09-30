"""Tests for middleware chaining utilities."""

import pytest

from ccproxy.core.middleware import BaseMiddleware, CompositeMiddleware, MiddlewareChain
from ccproxy.core.types import ProxyMethod, ProxyRequest, ProxyResponse


class RecordingMiddleware(BaseMiddleware):
    def __init__(self, name: str, log: list[str]):
        self.name = name
        self.log = log

    async def __call__(self, request, next):
        self.log.append(f"before:{self.name}")
        response = await next(request)
        response.metadata.setdefault("visited", []).append(self.name)
        self.log.append(f"after:{self.name}")
        return response


async def _terminal_handler(request):
    return ProxyResponse(status_code=200, metadata={"visited": ["handler"]})


def _make_request() -> ProxyRequest:
    return ProxyRequest(method=ProxyMethod.GET, url="https://example.com")


@pytest.mark.asyncio
async def test_middleware_chain_executes_in_order():
    log: list[str] = []
    m1 = RecordingMiddleware("one", log)
    m2 = RecordingMiddleware("two", log)

    chain = MiddlewareChain([m1, m2])
    response = await chain(_make_request(), _terminal_handler)

    assert response.metadata["visited"] == ["handler", "two", "one"]
    assert log == [
        "before:one",
        "before:two",
        "after:two",
        "after:one",
    ]


@pytest.mark.asyncio
async def test_composite_middleware_delegates_to_chain():
    log: list[str] = []
    composite = CompositeMiddleware([RecordingMiddleware("inner", log)])

    response = await composite(_make_request(), _terminal_handler)

    assert "inner" in response.metadata["visited"]
    assert log == ["before:inner", "after:inner"]
