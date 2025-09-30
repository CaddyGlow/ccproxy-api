"""Tests for the Pushgateway client and circuit breaker."""

import sys
import time
import types

from ccproxy.plugins.metrics import pushgateway
from ccproxy.plugins.metrics.config import MetricsConfig


if "prometheus_client" not in sys.modules:
    prom_module = types.ModuleType("prometheus_client")
    exposition = types.ModuleType("prometheus_client.exposition")
    exposition.generate_latest = lambda registry: b"metric 1"  # type: ignore[attr-defined]
    prom_module.exposition = exposition
    prom_module.CollectorRegistry = object  # type: ignore[attr-defined]
    sys.modules["prometheus_client"] = prom_module
    sys.modules["prometheus_client.exposition"] = exposition


class DummyRegistry:
    def collect(self):
        return []


def test_circuit_breaker_transitions(monkeypatch):
    cb = pushgateway.CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    assert cb.can_execute() is True

    cb.record_failure()
    assert cb.state == "CLOSED"
    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.can_execute() is False

    monkeypatch.setattr(time, "time", lambda: cb.last_failure_time + 2)
    assert cb.can_execute() is True
    assert cb.state == "HALF_OPEN"

    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0


def make_client(monkeypatch, url="http://localhost:9091"):
    monkeypatch.setattr(pushgateway, "PROMETHEUS_AVAILABLE", True)
    client = pushgateway.PushgatewayClient(
        MetricsConfig(pushgateway_enabled=True, pushgateway_url=url)
    )
    return client


def test_push_metrics_standard(monkeypatch):
    calls = []

    def fake_push_to_gateway(*args, **kwargs):
        calls.append(("push", args, kwargs))

    monkeypatch.setattr(pushgateway, "push_to_gateway", fake_push_to_gateway)
    monkeypatch.setattr(pushgateway, "pushadd_to_gateway", fake_push_to_gateway)
    monkeypatch.setattr(pushgateway, "delete_from_gateway", fake_push_to_gateway)

    client = make_client(monkeypatch)
    registry = DummyRegistry()

    assert client.push_metrics(registry, method="push") is True
    assert calls[0][0] == "push"

    assert client.push_add_metrics(registry) is True
    assert calls[1][0] == "push"


def test_push_metrics_remote_write(monkeypatch):
    responses = []

    class FakeResponse:
        def __init__(self, status_code=200, text=""):
            self.status_code = status_code
            self.text = text

    def fake_post(url, **kwargs):
        responses.append((url, kwargs))
        return FakeResponse(200)

    monkeypatch.setattr(pushgateway.httpx, "post", fake_post)
    monkeypatch.setattr(pushgateway, "push_to_gateway", lambda *a, **k: None)
    monkeypatch.setattr(pushgateway, "pushadd_to_gateway", lambda *a, **k: None)
    monkeypatch.setattr(pushgateway, "delete_from_gateway", lambda *a, **k: None)
    monkeypatch.setattr(pushgateway, "PROMETHEUS_AVAILABLE", True)

    client = pushgateway.PushgatewayClient(
        MetricsConfig(
            pushgateway_enabled=True, pushgateway_url="http://localhost/api/v1/write"
        )
    )

    registry = DummyRegistry()
    assert client.push_metrics(registry, method="push") is True
    assert responses[0][0].endswith("/api/v1/import/prometheus")


def test_push_metrics_disabled(monkeypatch):
    monkeypatch.setattr(pushgateway, "PROMETHEUS_AVAILABLE", True)
    client = pushgateway.PushgatewayClient(
        MetricsConfig(pushgateway_enabled=False, pushgateway_url="http://localhost")
    )
    assert client.push_metrics(DummyRegistry()) is False


def test_delete_metrics_remote_write(monkeypatch):
    monkeypatch.setattr(pushgateway, "PROMETHEUS_AVAILABLE", True)
    client = pushgateway.PushgatewayClient(
        MetricsConfig(
            pushgateway_enabled=True,
            pushgateway_url="http://localhost/api/v1/write",
        )
    )

    assert client.delete_metrics() is False
