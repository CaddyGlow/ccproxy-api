"""Testing utilities and mock response generation for CCProxy.

This package provides comprehensive testing utilities including:
- Mock response generation for bypass mode
- Request payload builders for dual-format testing
- Response processing and metrics collection
- Traffic pattern generation and scenario management
"""

from ccproxy.testing.config import (
    MockResponseConfig,
    RequestScenario,
    TrafficConfig,
    TrafficMetrics,
)
from ccproxy.testing.content_generation import MessageContentGenerator, PayloadBuilder
from ccproxy.testing.mock_responses import RealisticMockResponseGenerator
from ccproxy.testing.response_handlers import MetricsExtractor, ResponseHandler
from ccproxy.testing.scenarios import ScenarioGenerator, TrafficPatternAnalyzer


__all__ = [
    "MockResponseConfig",
    "RequestScenario",
    "TrafficConfig",
    "TrafficMetrics",
    "MessageContentGenerator",
    "PayloadBuilder",
    "RealisticMockResponseGenerator",
    "MetricsExtractor",
    "ResponseHandler",
    "ScenarioGenerator",
    "TrafficPatternAnalyzer",
]
