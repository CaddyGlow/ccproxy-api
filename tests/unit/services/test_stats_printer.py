"""Tests for stats printer functionality."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ccproxy.config.observability import ObservabilitySettings
from ccproxy.observability.stats_printer import (
    StatsCollector,
    StatsSnapshot,
    get_stats_collector,
    reset_stats_collector,
)


class TestStatsSnapshot:
    """Test StatsSnapshot dataclass."""

    def test_stats_snapshot_creation(self) -> None:
        """Test creating a StatsSnapshot."""
        timestamp = datetime.now()
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.requests_total == 100
        assert snapshot.requests_last_minute == 5
        assert snapshot.avg_response_time_ms == 150.5
        assert snapshot.avg_response_time_last_minute_ms == 200.0
        assert snapshot.tokens_input_total == 1000
        assert snapshot.tokens_output_total == 800
        assert snapshot.tokens_input_last_minute == 50
        assert snapshot.tokens_output_last_minute == 40
        assert snapshot.cost_total_usd == 1.25
        assert snapshot.cost_last_minute_usd == 0.05
        assert snapshot.errors_total == 2
        assert snapshot.errors_last_minute == 0
        assert snapshot.active_requests == 3
        assert snapshot.top_model == "claude-3-sonnet"
        assert snapshot.top_model_percentage == 75.0


class TestStatsCollector:
    """Test StatsCollector class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_interval=60.0,
            stats_printing_format="console",
        )
        self.mock_metrics = Mock()
        self.mock_metrics.is_enabled.return_value = True
        self.mock_storage = AsyncMock()
        self.mock_storage.is_enabled.return_value = True

    def test_stats_collector_initialization(self) -> None:
        """Test StatsCollector initialization."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=self.mock_metrics,
            storage_instance=self.mock_storage,
        )

        assert collector.settings == self.settings
        assert collector._metrics_instance == self.mock_metrics
        assert collector._storage_instance == self.mock_storage
        assert collector._last_snapshot is None

    @pytest.mark.asyncio
    async def test_collect_stats_default_values(self) -> None:
        """Test collecting stats with default values when no data available."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        snapshot = await collector.collect_stats()

        assert isinstance(snapshot, StatsSnapshot)
        assert snapshot.requests_total == 0
        assert snapshot.requests_last_minute == 0
        assert snapshot.avg_response_time_ms == 0.0
        assert snapshot.avg_response_time_last_minute_ms == 0.0
        assert snapshot.tokens_input_total == 0
        assert snapshot.tokens_output_total == 0
        assert snapshot.tokens_input_last_minute == 0
        assert snapshot.tokens_output_last_minute == 0
        assert snapshot.cost_total_usd == 0.0
        assert snapshot.cost_last_minute_usd == 0.0
        assert snapshot.errors_total == 0
        assert snapshot.errors_last_minute == 0
        assert snapshot.active_requests == 0
        assert snapshot.top_model == "unknown"
        assert snapshot.top_model_percentage == 0.0

    @pytest.mark.asyncio
    async def test_collect_from_prometheus(self) -> None:
        """Test collecting stats from Prometheus metrics."""
        # Mock Prometheus active requests gauge
        mock_active_requests = Mock()
        mock_active_requests._value._value = 5
        self.mock_metrics.active_requests = mock_active_requests

        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=self.mock_metrics,
            storage_instance=None,
        )

        snapshot = await collector.collect_stats()

        assert snapshot.active_requests == 5

    @pytest.mark.asyncio
    async def test_collect_from_duckdb(self) -> None:
        """Test collecting stats from DuckDB storage."""
        # Mock DuckDB analytics responses
        overall_analytics = {
            "summary": {
                "total_requests": 100,
                "avg_duration_ms": 150.5,
                "total_tokens_input": 1000,
                "total_tokens_output": 800,
                "total_cost_usd": 1.25,
            }
        }

        last_minute_analytics = {
            "summary": {
                "total_requests": 5,
                "avg_duration_ms": 200.0,
                "total_tokens_input": 50,
                "total_tokens_output": 40,
                "total_cost_usd": 0.05,
            }
        }

        top_model_results = [{"model": "claude-3-sonnet", "request_count": 4}]

        self.mock_storage.get_analytics.side_effect = [
            overall_analytics,
            last_minute_analytics,
        ]
        self.mock_storage.query.return_value = top_model_results

        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=self.mock_storage,
        )

        snapshot = await collector.collect_stats()

        assert snapshot.requests_total == 100
        assert snapshot.requests_last_minute == 5
        assert snapshot.avg_response_time_ms == 150.5
        assert snapshot.avg_response_time_last_minute_ms == 200.0
        assert snapshot.tokens_input_total == 1000
        assert snapshot.tokens_output_total == 800
        assert snapshot.tokens_input_last_minute == 50
        assert snapshot.tokens_output_last_minute == 40
        assert snapshot.cost_total_usd == 1.25
        assert snapshot.cost_last_minute_usd == 0.05
        assert snapshot.top_model == "claude-3-sonnet"
        assert snapshot.top_model_percentage == 80.0  # 4/5 * 100

    @pytest.mark.asyncio
    async def test_collect_from_duckdb_with_errors(self) -> None:
        """Test collecting stats from DuckDB with errors."""
        self.mock_storage.get_analytics.side_effect = Exception("Database error")

        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=self.mock_storage,
        )

        # Should not raise exception, should return default values
        snapshot = await collector.collect_stats()

        assert snapshot.requests_total == 0
        assert snapshot.requests_last_minute == 0

    def test_format_stats_console(self) -> None:
        """Test formatting stats for console output."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        formatted = collector.format_stats(snapshot)

        assert "[2024-01-01 12:00:00] METRICS SUMMARY" in formatted
        assert "Requests: 5 (last min) / 100 (total)" in formatted
        assert "Avg Response: 200.0ms (last min) / 150.5ms (overall)" in formatted
        assert "Tokens: 50 in / 40 out (last min)" in formatted
        assert "Cost: $0.0500 (last min) / $1.2500 (total)" in formatted
        assert "Errors: 0 (last min) / 2 (total)" in formatted
        assert "Active: 3 requests" in formatted
        assert "Top Model: claude-3-sonnet (75.0%)" in formatted

    def test_format_stats_json(self) -> None:
        """Test formatting stats for JSON output."""
        settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_format="json",
        )
        collector = StatsCollector(
            settings=settings,
            metrics_instance=None,
            storage_instance=None,
        )

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        formatted = collector.format_stats(snapshot)
        data = json.loads(formatted)

        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["requests"]["last_minute"] == 5
        assert data["requests"]["total"] == 100
        assert data["response_time_ms"]["last_minute"] == 200.0
        assert data["response_time_ms"]["overall"] == 150.5
        assert data["tokens"]["input_last_minute"] == 50
        assert data["tokens"]["output_last_minute"] == 40
        assert data["tokens"]["input_total"] == 1000
        assert data["tokens"]["output_total"] == 800
        assert data["cost_usd"]["last_minute"] == 0.05
        assert data["cost_usd"]["total"] == 1.25
        assert data["errors"]["last_minute"] == 0
        assert data["errors"]["total"] == 2
        assert data["active_requests"] == 3
        assert data["top_model"]["name"] == "claude-3-sonnet"
        assert data["top_model"]["percentage"] == 75.0

    def test_format_stats_rich(self) -> None:
        """Test formatting stats for rich output."""
        settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_format="rich",
        )
        collector = StatsCollector(
            settings=settings,
            metrics_instance=None,
            storage_instance=None,
        )

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        formatted = collector.format_stats(snapshot)

        # Check that it contains rich formatting elements or falls back to console
        assert "METRICS SUMMARY" in formatted
        assert "Requests" in formatted
        assert "5" in formatted  # requests_last_minute
        assert "100" in formatted  # requests_total
        assert "200.0ms" in formatted  # avg_response_time_last_minute_ms
        assert "150.5ms" in formatted  # avg_response_time_ms
        assert "claude-3-sonnet" in formatted
        assert "75.0%" in formatted

    def test_format_stats_log(self) -> None:
        """Test formatting stats for log output."""
        settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_format="log",
        )
        collector = StatsCollector(
            settings=settings,
            metrics_instance=None,
            storage_instance=None,
        )

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        formatted = collector.format_stats(snapshot)

        # Check that it contains log formatting elements
        assert "[2024-01-01 12:00:00]" in formatted
        assert "event=metrics_summary" in formatted
        assert "requests_last_min=5" in formatted
        assert "requests_total=100" in formatted
        assert "avg_response_ms=150.5" in formatted
        assert "avg_response_last_min_ms=200.0" in formatted
        assert "tokens_in_last_min=50" in formatted
        assert "tokens_out_last_min=40" in formatted
        assert "tokens_in_total=1000" in formatted
        assert "tokens_out_total=800" in formatted
        assert "cost_last_min_usd=0.0500" in formatted
        assert "cost_total_usd=1.2500" in formatted
        assert "errors_last_min=0" in formatted
        assert "errors_total=2" in formatted
        assert "active_requests=3" in formatted
        assert "top_model=claude-3-sonnet" in formatted
        assert "top_model_pct=75.0" in formatted

    def test_format_stats_default_fallback(self) -> None:
        """Test that unknown formats fall back to console."""
        settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_format="console",  # Will be validated, but we can test default path
        )
        collector = StatsCollector(
            settings=settings,
            metrics_instance=None,
            storage_instance=None,
        )

        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        snapshot = StatsSnapshot(
            timestamp=timestamp,
            requests_total=100,
            requests_last_minute=5,
            avg_response_time_ms=150.5,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=2,
            errors_last_minute=0,
            active_requests=3,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        formatted = collector.format_stats(snapshot)

        # Should format as console (default)
        assert "[2024-01-01 12:00:00] METRICS SUMMARY" in formatted
        assert "├─ Requests: 5 (last min) / 100 (total)" in formatted
        assert "├─ Avg Response: 200.0ms (last min) / 150.5ms (overall)" in formatted

    @pytest.mark.asyncio
    async def test_print_stats(self, capsys: Any) -> None:
        """Test printing stats to stdout."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        # Mock collect_stats to return a snapshot with meaningful activity
        with patch.object(collector, "collect_stats") as mock_collect:
            mock_collect.return_value = StatsSnapshot(
                timestamp=datetime.now(),
                requests_total=10,
                requests_last_minute=5,  # Meaningful activity
                avg_response_time_ms=150.0,
                avg_response_time_last_minute_ms=200.0,
                tokens_input_total=1000,
                tokens_output_total=800,
                tokens_input_last_minute=50,
                tokens_output_last_minute=40,
                cost_total_usd=1.25,
                cost_last_minute_usd=0.05,
                errors_total=0,
                errors_last_minute=0,
                active_requests=0,
                top_model="claude-3-sonnet",
                top_model_percentage=75.0,
            )

            await collector.print_stats()

        captured = capsys.readouterr()
        assert "METRICS SUMMARY" in captured.out
        assert "Requests:" in captured.out
        assert "Avg Response:" in captured.out
        assert "Tokens:" in captured.out
        assert "Cost:" in captured.out
        assert "Errors:" in captured.out
        assert "Active:" in captured.out
        assert "Top Model:" in captured.out

    @pytest.mark.asyncio
    async def test_print_stats_with_error(self, capsys: Any) -> None:
        """Test printing stats with error handling."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        # Mock collect_stats to raise exception
        with patch.object(
            collector, "collect_stats", side_effect=Exception("Test error")
        ):
            await collector.print_stats()

        # Should not raise exception, should log error
        captured = capsys.readouterr()
        assert captured.out == ""  # No output to stdout due to error

    def test_has_meaningful_activity_with_requests_last_minute(self) -> None:
        """Test meaningful activity detection with requests in last minute."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=100,
            requests_last_minute=5,  # Should trigger meaningful activity
            avg_response_time_ms=150.0,
            avg_response_time_last_minute_ms=200.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=50,
            tokens_output_last_minute=40,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.05,
            errors_total=0,
            errors_last_minute=0,
            active_requests=0,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        assert collector._has_meaningful_activity(snapshot) is True

    def test_has_meaningful_activity_with_active_requests(self) -> None:
        """Test meaningful activity detection with active requests."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=100,
            requests_last_minute=0,
            avg_response_time_ms=150.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.0,
            errors_total=0,
            errors_last_minute=0,
            active_requests=3,  # Should trigger meaningful activity
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        assert collector._has_meaningful_activity(snapshot) is True

    def test_has_meaningful_activity_with_errors_last_minute(self) -> None:
        """Test meaningful activity detection with errors in last minute."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=100,
            requests_last_minute=0,
            avg_response_time_ms=150.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.0,
            errors_total=2,
            errors_last_minute=1,  # Should trigger meaningful activity
            active_requests=0,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        assert collector._has_meaningful_activity(snapshot) is True

    def test_has_meaningful_activity_first_time_with_requests(self) -> None:
        """Test meaningful activity detection for first time with total requests."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )
        # No last snapshot (first time)
        assert collector._last_snapshot is None

        snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=100,  # Should trigger meaningful activity first time
            requests_last_minute=0,
            avg_response_time_ms=150.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=1000,
            tokens_output_total=800,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=1.25,
            cost_last_minute_usd=0.0,
            errors_total=0,
            errors_last_minute=0,
            active_requests=0,
            top_model="claude-3-sonnet",
            top_model_percentage=75.0,
        )

        assert collector._has_meaningful_activity(snapshot) is True

    def test_has_meaningful_activity_no_activity(self) -> None:
        """Test meaningful activity detection with no activity."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )
        # Simulate having a previous snapshot
        collector._last_snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=0,
            requests_last_minute=0,
            avg_response_time_ms=0.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=0,
            tokens_output_total=0,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=0.0,
            cost_last_minute_usd=0.0,
            errors_total=0,
            errors_last_minute=0,
            active_requests=0,
            top_model="unknown",
            top_model_percentage=0.0,
        )

        snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=0,
            requests_last_minute=0,
            avg_response_time_ms=0.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=0,
            tokens_output_total=0,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=0.0,
            cost_last_minute_usd=0.0,
            errors_total=0,
            errors_last_minute=0,
            active_requests=0,
            top_model="unknown",
            top_model_percentage=0.0,
        )

        assert collector._has_meaningful_activity(snapshot) is False

    @pytest.mark.asyncio
    async def test_print_stats_skipped_no_activity(self, capsys: Any) -> None:
        """Test that stats are skipped when there's no meaningful activity."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )
        # Simulate having a previous snapshot
        collector._last_snapshot = StatsSnapshot(
            timestamp=datetime.now(),
            requests_total=0,
            requests_last_minute=0,
            avg_response_time_ms=0.0,
            avg_response_time_last_minute_ms=0.0,
            tokens_input_total=0,
            tokens_output_total=0,
            tokens_input_last_minute=0,
            tokens_output_last_minute=0,
            cost_total_usd=0.0,
            cost_last_minute_usd=0.0,
            errors_total=0,
            errors_last_minute=0,
            active_requests=0,
            top_model="unknown",
            top_model_percentage=0.0,
        )

        await collector.print_stats()

        captured = capsys.readouterr()
        assert captured.out == ""  # No output to stdout due to no activity

    @pytest.mark.asyncio
    async def test_print_stats_with_meaningful_activity(self, capsys: Any) -> None:
        """Test that stats are printed when there's meaningful activity."""
        collector = StatsCollector(
            settings=self.settings,
            metrics_instance=None,
            storage_instance=None,
        )

        # Mock collect_stats to return a snapshot with activity
        with patch.object(collector, "collect_stats") as mock_collect:
            mock_collect.return_value = StatsSnapshot(
                timestamp=datetime.now(),
                requests_total=100,
                requests_last_minute=5,  # Meaningful activity
                avg_response_time_ms=150.0,
                avg_response_time_last_minute_ms=200.0,
                tokens_input_total=1000,
                tokens_output_total=800,
                tokens_input_last_minute=50,
                tokens_output_last_minute=40,
                cost_total_usd=1.25,
                cost_last_minute_usd=0.05,
                errors_total=0,
                errors_last_minute=0,
                active_requests=0,
                top_model="claude-3-sonnet",
                top_model_percentage=75.0,
            )

            await collector.print_stats()

        captured = capsys.readouterr()
        assert "METRICS SUMMARY" in captured.out
        assert "Requests:" in captured.out


class TestStatsCollectorGlobalFunctions:
    """Test global functions for stats collector."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_stats_collector()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        reset_stats_collector()

    def test_get_stats_collector_singleton(self) -> None:
        """Test that get_stats_collector returns singleton instance."""
        collector1 = get_stats_collector()
        collector2 = get_stats_collector()

        assert collector1 is collector2

    def test_reset_stats_collector(self) -> None:
        """Test resetting global stats collector."""
        collector1 = get_stats_collector()
        reset_stats_collector()
        collector2 = get_stats_collector()

        assert collector1 is not collector2

    def test_get_stats_collector_with_dependency_injection(self) -> None:
        """Test get_stats_collector with dependency injection."""
        settings = ObservabilitySettings(stats_printing_enabled=True)
        mock_metrics = Mock()
        mock_storage = Mock()

        collector = get_stats_collector(
            settings=settings,
            metrics_instance=mock_metrics,
            storage_instance=mock_storage,
        )

        assert collector.settings == settings
        assert collector._metrics_instance == mock_metrics
        assert collector._storage_instance == mock_storage

    @patch("ccproxy.observability.metrics.get_metrics")
    def test_get_stats_collector_with_metrics_error(
        self, mock_get_metrics: Any
    ) -> None:
        """Test get_stats_collector when metrics initialization fails."""
        mock_get_metrics.side_effect = Exception("Metrics error")

        collector = get_stats_collector()

        assert collector is not None
        assert collector._metrics_instance is None

    @patch("ccproxy.observability.storage.duckdb_simple.SimpleDuckDBStorage")
    def test_get_stats_collector_with_storage_error(
        self, mock_storage_class: Any
    ) -> None:
        """Test get_stats_collector when storage initialization fails."""
        mock_storage_class.side_effect = Exception("Storage error")

        collector = get_stats_collector()

        assert collector is not None
        assert collector._storage_instance is None


class TestStatsCollectorIntegration:
    """Integration tests for StatsCollector."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        reset_stats_collector()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        reset_stats_collector()

    @pytest.mark.asyncio
    async def test_end_to_end_stats_collection(self) -> None:
        """Test end-to-end stats collection with mocked components."""
        # Mock metrics instance
        mock_metrics = Mock()
        mock_metrics.is_enabled.return_value = True
        mock_active_requests = Mock()
        mock_active_requests._value._value = 5
        mock_metrics.active_requests = mock_active_requests

        # Mock storage instance
        mock_storage = AsyncMock()
        mock_storage.is_enabled.return_value = True
        mock_storage.get_analytics.side_effect = [
            {
                "summary": {
                    "total_requests": 100,
                    "avg_duration_ms": 150.5,
                    "total_tokens_input": 1000,
                    "total_tokens_output": 800,
                    "total_cost_usd": 1.25,
                }
            },
            {
                "summary": {
                    "total_requests": 5,
                    "avg_duration_ms": 200.0,
                    "total_tokens_input": 50,
                    "total_tokens_output": 40,
                    "total_cost_usd": 0.05,
                }
            },
        ]
        mock_storage.query.return_value = [
            {"model": "claude-3-sonnet", "request_count": 4}
        ]

        settings = ObservabilitySettings(
            stats_printing_enabled=True,
            stats_printing_interval=60.0,
            stats_printing_format="console",
        )

        collector = StatsCollector(
            settings=settings,
            metrics_instance=mock_metrics,
            storage_instance=mock_storage,
        )

        snapshot = await collector.collect_stats()

        # Verify all data is collected correctly
        assert snapshot.requests_total == 100
        assert snapshot.requests_last_minute == 5
        assert snapshot.avg_response_time_ms == 150.5
        assert snapshot.avg_response_time_last_minute_ms == 200.0
        assert snapshot.tokens_input_total == 1000
        assert snapshot.tokens_output_total == 800
        assert snapshot.tokens_input_last_minute == 50
        assert snapshot.tokens_output_last_minute == 40
        assert snapshot.cost_total_usd == 1.25
        assert snapshot.cost_last_minute_usd == 0.05
        assert snapshot.active_requests == 5
        assert snapshot.top_model == "claude-3-sonnet"
        assert snapshot.top_model_percentage == 80.0

        # Verify formatting works
        formatted = collector.format_stats(snapshot)
        assert "METRICS SUMMARY" in formatted
        assert "Requests: 5 (last min) / 100 (total)" in formatted
        assert "Active: 5 requests" in formatted
        assert "Top Model: claude-3-sonnet (80.0%)" in formatted
