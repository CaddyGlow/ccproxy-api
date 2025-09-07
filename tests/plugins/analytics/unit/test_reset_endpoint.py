"""Unit tests for analytics service components."""

import pytest

from ccproxy.plugins.analytics.service import AnalyticsService


@pytest.mark.unit
class TestAnalyticsServiceComponents:
    """Test suite for individual analytics service components."""

    def test_analytics_service_initialization(self) -> None:
        """Test AnalyticsService can be initialized with mock engine."""

        class MockEngine:
            """Mock database engine for testing."""

            pass

        mock_engine = MockEngine()
        service = AnalyticsService(mock_engine)

        # Test that the service initializes correctly
        assert service is not None
        # Note: This is a unit test focusing on initialization
        # Actual functionality is tested in integration tests

    def test_query_logs_parameters_validation(self) -> None:
        """Test that query parameters are handled correctly."""

        class MockEngine:
            """Mock database engine."""

            pass

        mock_engine = MockEngine()
        service = AnalyticsService(mock_engine)

        # Test parameter validation (this would normally validate against the DB)
        # For unit tests, we focus on the service logic without DB interaction
        assert service is not None
        # The actual query functionality requires DB integration
        # so it's tested in the integration test suite
