"""Tests for scenario generation utilities."""

from datetime import UTC, datetime, timedelta

import pytest

from ccproxy.testing.config import RequestScenario, TrafficConfig
from ccproxy.testing.scenarios import ScenarioGenerator, TrafficPatternAnalyzer


def _fixed_random_sequence(values):
    iterator = iter(values)

    def _next():
        try:
            return next(iterator)
        except StopIteration:  # pragma: no cover - defensive
            return 0.0

    return _next


def test_scenario_generator_basic(monkeypatch):
    config = TrafficConfig(
        duration_seconds=1,
        requests_per_second=1,
        pattern="constant",
        models=["model-a"],
        message_types=["chat"],
        streaming_probability=1.0,
        response_type="success",
        api_formats=["openai"],
        bypass_mode=False,
        real_api_keys={"openai": "sk-test"},
    )

    generator = ScenarioGenerator(config)

    monkeypatch.setattr("ccproxy.testing.scenarios.random.choice", lambda seq: seq[0])
    monkeypatch.setattr("ccproxy.testing.scenarios.random.random", lambda: 0.1)

    scenarios = generator.generate_scenarios()

    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario.model == "model-a"
    assert scenario.streaming is True
    assert scenario.api_format == "openai"
    assert scenario.headers["Authorization"].startswith("Bearer")
    assert scenario.headers["Accept"] == "text/event-stream"


def test_calculate_time_offset_burst(monkeypatch):
    config = TrafficConfig(
        duration_seconds=10,
        requests_per_second=10,
        pattern="burst",
        models=["m"],
        message_types=["t"],
        streaming_probability=0.0,
        response_type="success",
    )

    generator = ScenarioGenerator(config)
    offset_first = generator._calculate_time_offset(0, 20, config.duration_seconds)
    offset_after_burst = generator._calculate_time_offset(
        5, 20, config.duration_seconds
    )

    assert offset_first.total_seconds() == 0.0
    assert offset_after_burst > offset_first


def test_analyze_distribution():
    now = datetime.now(UTC)
    scenarios = [
        RequestScenario(
            model="m1",
            message_type="chat",
            streaming=True,
            response_type="success",
            timestamp=now,
            api_format="openai",
            headers={},
        ),
        RequestScenario(
            model="m2",
            message_type="tool",
            streaming=False,
            response_type="success",
            timestamp=now + timedelta(seconds=1),
            api_format="anthropic",
            headers={},
        ),
    ]

    analysis = TrafficPatternAnalyzer.analyze_distribution(scenarios)

    assert analysis["total_scenarios"] == 2
    assert pytest.approx(analysis["streaming_percentage"]) == 0.5
    assert set(analysis["api_format_distribution"]) == {"openai", "anthropic"}
