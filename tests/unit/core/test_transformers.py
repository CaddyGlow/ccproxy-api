"""Tests for transformer base classes."""

import pytest

from ccproxy.core.transformers import (
    BaseTransformer,
    RequestTransformer,
    ResponseTransformer,
)


class StubTransformer(BaseTransformer):
    async def transform(self, data, context=None):  # pragma: no cover - overridden
        return data


class EchoRequestTransformer(RequestTransformer):
    async def _transform_request(self, request, context=None):
        return {"payload": request, "context": context}


class EchoResponseTransformer(ResponseTransformer):
    async def _transform_response(self, response, context=None):
        return {"payload": response, "context": context}


@pytest.mark.asyncio
async def test_request_transformer_collects_metrics(monkeypatch):
    called = {}

    async def fake_collect(self, **kwargs):
        called.update(kwargs)

    t = EchoRequestTransformer()
    monkeypatch.setattr(
        t,
        "_collect_transformation_metrics",
        fake_collect.__get__(t, EchoRequestTransformer),
    )

    result = await t.transform({"hello": "world"}, context="ctx")

    assert result["payload"] == {"hello": "world"}
    assert called["transformation_type"] == "request"


@pytest.mark.asyncio
async def test_response_transformer_collects_metrics(monkeypatch):
    called = {}

    async def fake_collect(self, **kwargs):
        called.update(kwargs)

    t = EchoResponseTransformer()
    monkeypatch.setattr(
        t,
        "_collect_transformation_metrics",
        fake_collect.__get__(t, EchoResponseTransformer),
    )

    result = await t.transform("payload", context=None)

    assert result["payload"] == "payload"
    assert called["transformation_type"] == "response"


def test_transformer_calculates_size_for_various_types():
    base = StubTransformer()

    assert base._calculate_data_size(None) == 0
    assert base._calculate_data_size(b"abc") == 3
    assert base._calculate_data_size("abc") == 3
    assert base._calculate_data_size([1, 2]) >= 2
