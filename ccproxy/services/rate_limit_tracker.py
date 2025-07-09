"""Rate limit tracking and analytics service.

This module provides comprehensive rate limit tracking, analysis, and prediction
capabilities for both API key and OAuth authentication methods.
"""

import asyncio
import logging
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Optional

from ccproxy.config.settings import Settings
from ccproxy.models.rate_limit import (
    OAuthUnifiedRateLimit,
    RateLimitData,
    RateLimitStatus,
    StandardRateLimit,
)


logger = logging.getLogger(__name__)


class RateLimitDataPoint:
    """Single data point for rate limit tracking."""

    def __init__(
        self,
        timestamp: datetime,
        auth_type: str,
        requests_used: int = 0,
        requests_limit: int | None = None,
        tokens_used: int = 0,
        tokens_limit: int | None = None,
        utilization_percentage: float | None = None,
        reset_timestamp: datetime | None = None,
    ):
        self.timestamp = timestamp
        self.auth_type = auth_type
        self.requests_used = requests_used
        self.requests_limit = requests_limit
        self.tokens_used = tokens_used
        self.tokens_limit = tokens_limit
        self.utilization_percentage = utilization_percentage
        self.reset_timestamp = reset_timestamp


class RateLimitTracker:
    """Thread-safe rate limit tracker with analytics and prediction capabilities."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the rate limit tracker.

        Args:
            settings: Configuration settings instance
        """
        self.settings = settings or Settings()
        self._lock = threading.RLock()

        # In-memory storage with TTL
        self._data_cache: dict[str, deque[RateLimitDataPoint]] = defaultdict(deque)
        self._current_status: dict[str, RateLimitStatus] = {}
        self._last_cleanup = datetime.now()

        # Configuration
        self._cache_ttl = self.settings.rate_limit_cache_ttl
        self._prediction_window = self.settings.rate_limit_prediction_window
        self._alert_threshold = self.settings.rate_limit_alert_threshold
        self._max_data_points = 1000  # Maximum data points per auth type

        # Analytics tracking
        self._usage_trends: dict[str, dict[str, Any]] = defaultdict(dict)

        logger.info(
            f"RateLimitTracker initialized with cache_ttl={self._cache_ttl}s, "
            f"prediction_window={self._prediction_window}s, "
            f"alert_threshold={self._alert_threshold}"
        )

    def track_rate_limit(self, rate_limit_data: RateLimitData) -> None:
        """Store and track rate limit data.

        Args:
            rate_limit_data: Rate limit data to track
        """
        try:
            with self._lock:
                auth_type = rate_limit_data.auth_type
                timestamp = rate_limit_data.timestamp

                # Extract metrics based on auth type
                if auth_type == "api_key" and rate_limit_data.standard:
                    data_point = self._create_standard_data_point(
                        timestamp, auth_type, rate_limit_data.standard
                    )
                elif auth_type == "oauth" and rate_limit_data.oauth_unified:
                    data_point = self._create_oauth_data_point(
                        timestamp, auth_type, rate_limit_data.oauth_unified
                    )
                else:
                    logger.warning(
                        f"Unable to track rate limit data for auth_type={auth_type}: "
                        f"missing appropriate rate limit data"
                    )
                    return

                # Store data point
                self._data_cache[auth_type].append(data_point)

                # Maintain cache size limit
                if len(self._data_cache[auth_type]) > self._max_data_points:
                    self._data_cache[auth_type].popleft()

                # Update current status
                self._update_current_status(auth_type, data_point)

                # Update usage trends
                self._update_usage_trends(auth_type, data_point)

                # Periodic cleanup
                self._cleanup_expired_data()

                logger.debug(f"Tracked rate limit data for auth_type={auth_type}")

        except Exception as e:
            logger.error(f"Error tracking rate limit data: {e}")

    def get_current_status(self, auth_type: str) -> RateLimitStatus | None:
        """Get current rate limit status for auth type.

        Args:
            auth_type: Authentication type to check

        Returns:
            Current rate limit status or None if not available
        """
        with self._lock:
            return self._current_status.get(auth_type)

    def calculate_utilization(self, auth_type: str) -> float:
        """Calculate current utilization percentage.

        Args:
            auth_type: Authentication type to calculate for

        Returns:
            Utilization percentage (0.0-100.0)
        """
        with self._lock:
            if auth_type not in self._data_cache:
                return 0.0

            data_points = self._data_cache[auth_type]
            if not data_points:
                return 0.0

            latest_point = data_points[-1]
            if latest_point.utilization_percentage is not None:
                return latest_point.utilization_percentage

            # Calculate based on available data
            if auth_type == "api_key":
                # For API keys, use requests or tokens (whichever is more limiting)
                request_util = 0.0
                token_util = 0.0

                if latest_point.requests_limit and latest_point.requests_limit > 0:
                    request_util = (
                        latest_point.requests_used / latest_point.requests_limit
                    ) * 100

                if latest_point.tokens_limit and latest_point.tokens_limit > 0:
                    token_util = (
                        latest_point.tokens_used / latest_point.tokens_limit
                    ) * 100

                return max(request_util, token_util)

            return 0.0

    def predict_exhaustion_time(self, auth_type: str) -> datetime | None:
        """Predict when rate limits will be exhausted.

        Args:
            auth_type: Authentication type to predict for

        Returns:
            Estimated exhaustion time or None if cannot predict
        """
        with self._lock:
            if auth_type not in self._data_cache:
                return None

            data_points = list(self._data_cache[auth_type])
            if len(data_points) < 2:
                return None

            now = datetime.now()
            cutoff_time = now - timedelta(seconds=self._prediction_window)

            # Filter recent data points
            recent_points = [p for p in data_points if p.timestamp >= cutoff_time]

            if len(recent_points) < 2:
                return None

            try:
                # Calculate usage trend
                usage_trend = self._calculate_usage_trend(recent_points)
                if usage_trend <= 0:
                    return None  # No increasing usage trend

                latest_point = recent_points[-1]
                current_utilization = self.calculate_utilization(auth_type)

                if current_utilization >= 100:
                    return now  # Already exhausted

                # Calculate time to exhaustion
                remaining_percentage = 100 - current_utilization
                seconds_to_exhaustion = remaining_percentage / usage_trend

                exhaustion_time = now + timedelta(seconds=seconds_to_exhaustion)

                # Cap prediction to reset time if available
                if latest_point.reset_timestamp:
                    exhaustion_time = min(exhaustion_time, latest_point.reset_timestamp)

                return exhaustion_time

            except Exception as e:
                logger.error(f"Error predicting exhaustion time: {e}")
                return None

    def is_approaching_limit(self, auth_type: str, threshold: float = 0.8) -> bool:
        """Check if rate limit is approaching the threshold.

        Args:
            auth_type: Authentication type to check
            threshold: Threshold percentage (0.0-1.0)

        Returns:
            True if approaching limit, False otherwise
        """
        utilization = self.calculate_utilization(auth_type)
        return utilization >= (threshold * 100)

    def get_usage_statistics(self, auth_type: str) -> dict[str, Any]:
        """Get comprehensive usage statistics.

        Args:
            auth_type: Authentication type to get statistics for

        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            if auth_type not in self._data_cache:
                return {}

            data_points = list(self._data_cache[auth_type])
            if not data_points:
                return {}

            now = datetime.now()
            recent_points = [
                p
                for p in data_points
                if p.timestamp >= now - timedelta(seconds=self._prediction_window)
            ]

            stats = {
                "auth_type": auth_type,
                "total_data_points": len(data_points),
                "recent_data_points": len(recent_points),
                "current_utilization": self.calculate_utilization(auth_type),
                "is_approaching_limit": self.is_approaching_limit(auth_type),
                "predicted_exhaustion_time": self.predict_exhaustion_time(auth_type),
                "usage_trends": self._usage_trends.get(auth_type, {}),
            }

            if recent_points:
                latest_point = recent_points[-1]
                stats.update(
                    {
                        "last_update": latest_point.timestamp.isoformat(),
                        "reset_timestamp": (
                            latest_point.reset_timestamp.isoformat()
                            if latest_point.reset_timestamp
                            else None
                        ),
                    }
                )

            return stats

    def clear_cache(self, auth_type: str | None = None) -> None:
        """Clear cached data.

        Args:
            auth_type: Specific auth type to clear, or None for all
        """
        with self._lock:
            if auth_type:
                self._data_cache.pop(auth_type, None)
                self._current_status.pop(auth_type, None)
                self._usage_trends.pop(auth_type, None)
                logger.info(f"Cleared cache for auth_type={auth_type}")
            else:
                self._data_cache.clear()
                self._current_status.clear()
                self._usage_trends.clear()
                logger.info("Cleared all cache data")

    def _create_standard_data_point(
        self, timestamp: datetime, auth_type: str, standard: StandardRateLimit
    ) -> RateLimitDataPoint:
        """Create data point from standard rate limit data."""
        requests_used = 0
        if standard.requests_limit and standard.requests_remaining:
            requests_used = standard.requests_limit - standard.requests_remaining

        tokens_used = 0
        if standard.tokens_limit and standard.tokens_remaining:
            tokens_used = standard.tokens_limit - standard.tokens_remaining

        return RateLimitDataPoint(
            timestamp=timestamp,
            auth_type=auth_type,
            requests_used=requests_used,
            requests_limit=standard.requests_limit,
            tokens_used=tokens_used,
            tokens_limit=standard.tokens_limit,
            reset_timestamp=standard.reset_timestamp,
        )

    def _create_oauth_data_point(
        self, timestamp: datetime, auth_type: str, oauth: OAuthUnifiedRateLimit
    ) -> RateLimitDataPoint:
        """Create data point from OAuth rate limit data."""
        utilization_percentage = oauth.fallback_percentage

        return RateLimitDataPoint(
            timestamp=timestamp,
            auth_type=auth_type,
            utilization_percentage=utilization_percentage,
            reset_timestamp=oauth.reset_timestamp,
        )

    def _update_current_status(
        self, auth_type: str, data_point: RateLimitDataPoint
    ) -> None:
        """Update current rate limit status."""
        utilization = self.calculate_utilization(auth_type)
        is_limited = utilization >= 100

        time_until_reset = None
        if data_point.reset_timestamp:
            time_delta = data_point.reset_timestamp - datetime.now()
            time_until_reset = max(0, int(time_delta.total_seconds()))

        self._current_status[auth_type] = RateLimitStatus(
            auth_type=auth_type,
            is_limited=is_limited,
            utilization_percentage=utilization,
            time_until_reset=time_until_reset,
            estimated_exhaustion_time=self.predict_exhaustion_time(auth_type),
        )

    def _update_usage_trends(
        self, auth_type: str, data_point: RateLimitDataPoint
    ) -> None:
        """Update usage trend calculations."""
        if auth_type not in self._usage_trends:
            self._usage_trends[auth_type] = {}

        now = datetime.now()
        trends = self._usage_trends[auth_type]

        # Calculate recent usage rate
        recent_points = [
            p
            for p in self._data_cache[auth_type]
            if p.timestamp >= now - timedelta(seconds=self._prediction_window)
        ]

        if len(recent_points) >= 2:
            trends["usage_rate"] = self._calculate_usage_trend(recent_points)

        # Update last seen metrics
        trends["last_utilization"] = data_point.utilization_percentage or 0.0
        trends["last_update"] = now.isoformat()

    def _calculate_usage_trend(self, data_points: list[RateLimitDataPoint]) -> float:
        """Calculate usage trend from data points."""
        if len(data_points) < 2:
            return 0.0

        # Simple linear trend calculation
        start_point = data_points[0]
        end_point = data_points[-1]

        time_diff = (end_point.timestamp - start_point.timestamp).total_seconds()
        if time_diff <= 0:
            return 0.0

        start_util = start_point.utilization_percentage or 0.0
        end_util = end_point.utilization_percentage or 0.0

        # Calculate utilization change per second
        usage_change = end_util - start_util
        return usage_change / time_diff

    def _cleanup_expired_data(self) -> None:
        """Clean up expired data points."""
        now = datetime.now()

        # Only cleanup periodically to avoid performance impact
        if (now - self._last_cleanup).total_seconds() < 60:
            return

        cutoff_time = now - timedelta(seconds=self._cache_ttl * 10)  # Keep 10x TTL

        for auth_type in list(self._data_cache.keys()):
            data_points = self._data_cache[auth_type]

            # Remove expired data points
            while data_points and data_points[0].timestamp < cutoff_time:
                data_points.popleft()

            # Remove empty caches
            if not data_points:
                del self._data_cache[auth_type]
                self._current_status.pop(auth_type, None)
                self._usage_trends.pop(auth_type, None)

        self._last_cleanup = now
        logger.debug("Completed rate limit data cleanup")


# Global instance for application-wide usage
_rate_limit_tracker: RateLimitTracker | None = None


def get_rate_limit_tracker(settings: Settings | None = None) -> RateLimitTracker:
    """Get global rate limit tracker instance.

    Args:
        settings: Configuration settings (used only for first initialization)

    Returns:
        Global RateLimitTracker instance
    """
    global _rate_limit_tracker

    if _rate_limit_tracker is None:
        _rate_limit_tracker = RateLimitTracker(settings)

    return _rate_limit_tracker


async def track_rate_limit_async(rate_limit_data: RateLimitData) -> None:
    """Async wrapper for tracking rate limit data.

    Args:
        rate_limit_data: Rate limit data to track
    """
    loop = asyncio.get_event_loop()
    tracker = get_rate_limit_tracker()

    await loop.run_in_executor(None, tracker.track_rate_limit, rate_limit_data)


__all__ = [
    "RateLimitTracker",
    "RateLimitDataPoint",
    "get_rate_limit_tracker",
    "track_rate_limit_async",
]
