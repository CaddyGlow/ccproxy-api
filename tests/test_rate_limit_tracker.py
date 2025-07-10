"""Unit tests for rate limit tracker service."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from ccproxy.config.settings import Settings
from ccproxy.models.rate_limit import (
    OAuthUnifiedRateLimit,
    RateLimitData,
    RateLimitStatus,
    StandardRateLimit,
)
from ccproxy.services.rate_limit_tracker import (
    RateLimitDataPoint,
    RateLimitTracker,
    get_rate_limit_tracker,
    track_rate_limit_async,
)


class TestRateLimitDataPoint:
    """Test the RateLimitDataPoint class."""

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        timestamp = datetime.now()
        point = RateLimitDataPoint(timestamp=timestamp, auth_type="api_key")

        assert point.timestamp == timestamp
        assert point.auth_type == "api_key"
        assert point.requests_used == 0
        assert point.requests_limit is None
        assert point.tokens_used == 0
        assert point.tokens_limit is None
        assert point.utilization_percentage is None
        assert point.reset_timestamp is None

    def test_init_full(self):
        """Test initialization with all parameters."""
        timestamp = datetime.now()
        reset_timestamp = timestamp + timedelta(hours=1)

        point = RateLimitDataPoint(
            timestamp=timestamp,
            auth_type="api_key",
            requests_used=100,
            requests_limit=1000,
            tokens_used=5000,
            tokens_limit=10000,
            utilization_percentage=50.0,
            reset_timestamp=reset_timestamp,
        )

        assert point.timestamp == timestamp
        assert point.auth_type == "api_key"
        assert point.requests_used == 100
        assert point.requests_limit == 1000
        assert point.tokens_used == 5000
        assert point.tokens_limit == 10000
        assert point.utilization_percentage == 50.0
        assert point.reset_timestamp == reset_timestamp


class TestRateLimitTracker:
    """Test the RateLimitTracker class."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.rate_limit_cache_ttl = 30
        settings.rate_limit_prediction_window = 300
        settings.rate_limit_alert_threshold = 0.8
        return settings

    @pytest.fixture
    def tracker(self, mock_settings):
        """Create a tracker instance for testing."""
        return RateLimitTracker(mock_settings)

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        with patch(
            "ccproxy.services.rate_limit_tracker.Settings"
        ) as mock_settings_class:
            mock_settings = Mock()
            mock_settings.rate_limit_cache_ttl = 30
            mock_settings.rate_limit_prediction_window = 300
            mock_settings.rate_limit_alert_threshold = 0.8
            mock_settings_class.return_value = mock_settings

            tracker = RateLimitTracker()

            assert tracker.settings == mock_settings
            assert tracker._cache_ttl == 30
            assert tracker._prediction_window == 300
            assert tracker._alert_threshold == 0.8

    def test_init_custom_settings(self, mock_settings):
        """Test initialization with custom settings."""
        tracker = RateLimitTracker(mock_settings)

        assert tracker.settings == mock_settings
        assert tracker._cache_ttl == 30
        assert tracker._prediction_window == 300
        assert tracker._alert_threshold == 0.8

    def test_track_rate_limit_api_key(self, tracker):
        """Test tracking rate limit data for API key."""
        timestamp = datetime.now()
        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=800,
            tokens_limit=10000,
            tokens_remaining=7000,
            reset_timestamp=timestamp + timedelta(hours=1),
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Check that data was stored
        assert "api_key" in tracker._data_cache
        assert len(tracker._data_cache["api_key"]) == 1

        data_point = tracker._data_cache["api_key"][0]
        assert data_point.auth_type == "api_key"
        assert data_point.requests_used == 200  # 1000 - 800
        assert data_point.tokens_used == 3000  # 10000 - 7000
        assert data_point.requests_limit == 1000
        assert data_point.tokens_limit == 10000

        # Check current status was updated
        status = tracker.get_current_status("api_key")
        assert status is not None
        assert status.auth_type == "api_key"
        assert not status.is_limited

    def test_track_rate_limit_oauth(self, tracker):
        """Test tracking rate limit data for OAuth."""
        timestamp = datetime.now()
        oauth = OAuthUnifiedRateLimit(
            status="allowed",
            fallback_percentage=75.0,
            reset_timestamp=timestamp + timedelta(hours=1),
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Check that data was stored
        assert "oauth" in tracker._data_cache
        assert len(tracker._data_cache["oauth"]) == 1

        data_point = tracker._data_cache["oauth"][0]
        assert data_point.auth_type == "oauth"
        assert data_point.utilization_percentage == 75.0

        # Check current status was updated
        status = tracker.get_current_status("oauth")
        assert status is not None
        assert status.auth_type == "oauth"
        assert not status.is_limited

    def test_track_rate_limit_invalid_data(self, tracker):
        """Test tracking with invalid rate limit data."""
        timestamp = datetime.now()

        # API key without standard data
        rate_limit_data = RateLimitData(
            auth_type="api_key",
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Should not store invalid data
        assert "api_key" not in tracker._data_cache

    def test_get_current_status_not_found(self, tracker):
        """Test getting current status for non-existent auth type."""
        status = tracker.get_current_status("nonexistent")
        assert status is None

    def test_calculate_utilization_api_key(self, tracker):
        """Test utilization calculation for API key."""
        timestamp = datetime.now()
        standard = StandardRateLimit(
            requests_limit=1000,
            requests_remaining=600,  # 40% used
            tokens_limit=10000,
            tokens_remaining=3000,  # 70% used
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Should return the higher utilization (tokens at 70%)
        utilization = tracker.calculate_utilization("api_key")
        assert utilization == 70.0

    def test_calculate_utilization_oauth(self, tracker):
        """Test utilization calculation for OAuth."""
        timestamp = datetime.now()
        oauth = OAuthUnifiedRateLimit(
            fallback_percentage=85.0,
        )

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        utilization = tracker.calculate_utilization("oauth")
        assert utilization == 85.0

    def test_calculate_utilization_empty(self, tracker):
        """Test utilization calculation with no data."""
        utilization = tracker.calculate_utilization("nonexistent")
        assert utilization == 0.0

    def test_is_approaching_limit(self, tracker):
        """Test limit approach detection."""
        timestamp = datetime.now()

        # Create data with 90% utilization
        oauth = OAuthUnifiedRateLimit(fallback_percentage=90.0)
        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Should be approaching limit (90% > 80% threshold)
        assert tracker.is_approaching_limit("oauth")
        assert tracker.is_approaching_limit("oauth", threshold=0.8)
        assert not tracker.is_approaching_limit("oauth", threshold=0.95)

    def test_predict_exhaustion_time_insufficient_data(self, tracker):
        """Test exhaustion prediction with insufficient data."""
        timestamp = datetime.now()
        oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Should return None with only one data point
        exhaustion_time = tracker.predict_exhaustion_time("oauth")
        assert exhaustion_time is None

    def test_predict_exhaustion_time_with_trend(self, tracker):
        """Test exhaustion prediction with usage trend."""
        base_time = datetime.now()

        # Create multiple data points with increasing utilization
        for i, utilization in enumerate([50.0, 60.0, 70.0]):
            oauth = OAuthUnifiedRateLimit(fallback_percentage=utilization)
            rate_limit_data = RateLimitData(
                auth_type="oauth",
                oauth_unified=oauth,
                timestamp=base_time + timedelta(seconds=i * 60),
            )
            tracker.track_rate_limit(rate_limit_data)

        exhaustion_time = tracker.predict_exhaustion_time("oauth")

        # Should predict some time in the future
        assert exhaustion_time is not None
        assert exhaustion_time > base_time

    def test_predict_exhaustion_time_no_trend(self, tracker):
        """Test exhaustion prediction with no usage trend."""
        base_time = datetime.now()

        # Create multiple data points with same utilization
        for i in range(3):
            oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)
            rate_limit_data = RateLimitData(
                auth_type="oauth",
                oauth_unified=oauth,
                timestamp=base_time + timedelta(seconds=i * 60),
            )
            tracker.track_rate_limit(rate_limit_data)

        exhaustion_time = tracker.predict_exhaustion_time("oauth")

        # Should return None for flat trend
        assert exhaustion_time is None

    def test_get_usage_statistics(self, tracker):
        """Test usage statistics retrieval."""
        timestamp = datetime.now()
        oauth = OAuthUnifiedRateLimit(fallback_percentage=75.0)

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        stats = tracker.get_usage_statistics("oauth")

        assert stats["auth_type"] == "oauth"
        assert stats["total_data_points"] == 1
        assert stats["current_utilization"] == 75.0
        assert stats["is_approaching_limit"] is False
        assert "last_update" in stats

    def test_get_usage_statistics_empty(self, tracker):
        """Test usage statistics for non-existent auth type."""
        stats = tracker.get_usage_statistics("nonexistent")
        assert stats == {}

    def test_clear_cache_specific(self, tracker):
        """Test clearing cache for specific auth type."""
        timestamp = datetime.now()

        # Add data for both auth types
        for auth_type in ["api_key", "oauth"]:
            if auth_type == "api_key":
                data = RateLimitData(
                    auth_type=auth_type,  # type: ignore
                    standard=StandardRateLimit(
                        requests_limit=1000, requests_remaining=500
                    ),
                    timestamp=timestamp,
                )
            else:
                data = RateLimitData(
                    auth_type=auth_type,  # type: ignore
                    oauth_unified=OAuthUnifiedRateLimit(fallback_percentage=50.0),
                    timestamp=timestamp,
                )

            tracker.track_rate_limit(data)

        # Clear only api_key
        tracker.clear_cache("api_key")

        assert "api_key" not in tracker._data_cache
        assert "oauth" in tracker._data_cache

    def test_clear_cache_all(self, tracker):
        """Test clearing all cache data."""
        timestamp = datetime.now()

        # Add some data
        oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)
        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Clear all
        tracker.clear_cache()

        assert len(tracker._data_cache) == 0
        assert len(tracker._current_status) == 0
        assert len(tracker._usage_trends) == 0

    def test_thread_safety(self, tracker):
        """Test basic thread safety of the tracker."""
        import threading

        timestamp = datetime.now()
        results = []

        def track_data():
            oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)
            rate_limit_data = RateLimitData(
                auth_type="oauth",
                oauth_unified=oauth,
                timestamp=timestamp,
            )
            tracker.track_rate_limit(rate_limit_data)
            results.append(tracker.calculate_utilization("oauth"))

        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=track_data)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should have succeeded
        assert len(results) == 5
        assert all(result == 50.0 for result in results)

    def test_cache_size_limit(self, tracker):
        """Test that cache respects size limits."""
        # Override max data points for testing
        tracker._max_data_points = 3

        timestamp = datetime.now()

        # Add more data points than the limit
        for i in range(5):
            oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)
            rate_limit_data = RateLimitData(
                auth_type="oauth",
                oauth_unified=oauth,
                timestamp=timestamp + timedelta(seconds=i),
            )
            tracker.track_rate_limit(rate_limit_data)

        # Should only keep the last 3 data points
        assert len(tracker._data_cache["oauth"]) == 3

    def test_cleanup_expired_data(self, tracker):
        """Test cleanup of expired data."""
        # Create data that's older than 10x TTL (10 * 30 = 300 seconds)
        old_time = datetime.now() - timedelta(seconds=400)

        oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)
        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=old_time,
        )

        tracker.track_rate_limit(rate_limit_data)

        # Force cleanup by setting last cleanup time to past
        tracker._last_cleanup = old_time

        # Add new data to trigger cleanup
        new_oauth = OAuthUnifiedRateLimit(fallback_percentage=60.0)
        new_rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=new_oauth,
            timestamp=datetime.now(),
        )

        tracker.track_rate_limit(new_rate_limit_data)

        # Old data should be removed (older than 10x TTL)
        assert len(tracker._data_cache["oauth"]) == 1


class TestGlobalTracker:
    """Test global tracker functionality."""

    def test_get_rate_limit_tracker_singleton(self):
        """Test that get_rate_limit_tracker returns singleton."""
        # Reset global state
        import ccproxy.services.rate_limit_tracker

        ccproxy.services.rate_limit_tracker._rate_limit_tracker = None

        tracker1 = get_rate_limit_tracker()
        tracker2 = get_rate_limit_tracker()

        assert tracker1 is tracker2

    def test_get_rate_limit_tracker_with_settings(self):
        """Test get_rate_limit_tracker with custom settings."""
        # Reset global state
        import ccproxy.services.rate_limit_tracker

        ccproxy.services.rate_limit_tracker._rate_limit_tracker = None

        mock_settings = Mock(spec=Settings)
        mock_settings.rate_limit_cache_ttl = 60
        mock_settings.rate_limit_prediction_window = 600
        mock_settings.rate_limit_alert_threshold = 0.9

        tracker = get_rate_limit_tracker(mock_settings)

        assert tracker.settings == mock_settings
        assert tracker._cache_ttl == 60

    @pytest.mark.asyncio
    async def test_track_rate_limit_async(self):
        """Test async rate limit tracking."""
        timestamp = datetime.now()
        oauth = OAuthUnifiedRateLimit(fallback_percentage=50.0)

        rate_limit_data = RateLimitData(
            auth_type="oauth",
            oauth_unified=oauth,
            timestamp=timestamp,
        )

        # Reset global state
        import ccproxy.services.rate_limit_tracker

        ccproxy.services.rate_limit_tracker._rate_limit_tracker = None

        await track_rate_limit_async(rate_limit_data)

        # Check that data was tracked
        tracker = get_rate_limit_tracker()
        assert "oauth" in tracker._data_cache
        assert len(tracker._data_cache["oauth"]) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_track_rate_limit_with_exception(self):
        """Test tracking with exception during processing."""
        mock_settings = Mock(spec=Settings)
        mock_settings.rate_limit_cache_ttl = 30
        mock_settings.rate_limit_prediction_window = 300
        mock_settings.rate_limit_alert_threshold = 0.8

        tracker = RateLimitTracker(mock_settings)

        # Mock an exception during processing
        with patch.object(
            tracker, "_create_standard_data_point", side_effect=Exception("Test error")
        ):
            timestamp = datetime.now()
            standard = StandardRateLimit(requests_limit=1000, requests_remaining=500)

            rate_limit_data = RateLimitData(
                auth_type="api_key",
                standard=standard,
                timestamp=timestamp,
            )

            # Should not raise exception
            tracker.track_rate_limit(rate_limit_data)

            # Data should not be stored
            assert "api_key" not in tracker._data_cache

    def test_utilization_calculation_edge_cases(self):
        """Test utilization calculation edge cases."""
        mock_settings = Mock(spec=Settings)
        mock_settings.rate_limit_cache_ttl = 30
        mock_settings.rate_limit_prediction_window = 300
        mock_settings.rate_limit_alert_threshold = 0.8

        tracker = RateLimitTracker(mock_settings)

        # Test with zero limits
        timestamp = datetime.now()
        standard = StandardRateLimit(
            requests_limit=0,
            requests_remaining=0,
            tokens_limit=0,
            tokens_remaining=0,
        )

        rate_limit_data = RateLimitData(
            auth_type="api_key",
            standard=standard,
            timestamp=timestamp,
        )

        tracker.track_rate_limit(rate_limit_data)

        utilization = tracker.calculate_utilization("api_key")
        assert utilization == 0.0

    def test_prediction_with_reset_timestamp(self):
        """Test prediction capped by reset timestamp."""
        mock_settings = Mock(spec=Settings)
        mock_settings.rate_limit_cache_ttl = 30
        mock_settings.rate_limit_prediction_window = 300
        mock_settings.rate_limit_alert_threshold = 0.8

        tracker = RateLimitTracker(mock_settings)

        base_time = datetime.now()
        reset_time = base_time + timedelta(minutes=5)

        # Create trend that would predict exhaustion after reset time
        for i, utilization in enumerate([80.0, 85.0, 90.0]):
            oauth = OAuthUnifiedRateLimit(
                fallback_percentage=utilization,
                reset_timestamp=reset_time,
            )
            rate_limit_data = RateLimitData(
                auth_type="oauth",
                oauth_unified=oauth,
                timestamp=base_time + timedelta(seconds=i * 60),
            )
            tracker.track_rate_limit(rate_limit_data)

        exhaustion_time = tracker.predict_exhaustion_time("oauth")

        # Should be capped at reset time
        assert exhaustion_time is not None
        assert exhaustion_time <= reset_time
