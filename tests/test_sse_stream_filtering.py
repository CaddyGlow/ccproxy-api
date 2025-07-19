"""
Tests for SSE stream filtering functionality.

This module tests the GET /logs/stream endpoint with filtering capabilities
similar to analytics and entries endpoints.
"""

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx._types import QueryParamTypes


class TestSSEStreamFiltering:
    """Test suite for SSE stream filtering functionality."""

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_no_filters(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream without any filters."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        with client_no_auth.stream("GET", "/logs/stream") as response:
            assert response.status_code == 200
            assert (
                response.headers["content-type"] == "text/event-stream; charset=utf-8"
            )
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["connection"] == "keep-alive"

            # Should receive connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_model_filter(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with model filter."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"model": "claude-3-5-sonnet-20241022"}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should receive connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_service_type_filter(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with service type filter."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"service_type": "proxy_service"}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should receive connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_service_type_negation_filter(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with service type negation filter."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"service_type": "!access_log,!sdk_service"}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should still get connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_duration_filters(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with duration range filters."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"min_duration_ms": 100.0, "max_duration_ms": 500.0}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should still get connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_status_code_filters(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with status code range filters."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"status_code_min": 200, "status_code_max": 299}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should still get connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_with_multiple_filters(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with multiple combined filters."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "service_type": "proxy_service",
            "min_duration_ms": 50.0,
            "max_duration_ms": 1000.0,
            "status_code_min": 200,
            "status_code_max": 299,
        }

        with client_no_auth.stream(
            "GET", "/logs/stream", params=cast(QueryParamTypes, params)
        ) as response:
            assert response.status_code == 200

            # Should still get connection event
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_filters_request_complete_events(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test that filters are applied to request_complete events."""

        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock events that should be filtered
        async def mock_events(connection_id=None, request_id=None):
            events = [
                # Connection event (should always pass)
                'data: {"type": "connection", "message": "Connected"}\n\n',
                # Event that matches filters
                'data: {"type": "request_complete", "data": {"model": "claude-3-5-sonnet-20241022", "service_type": "proxy_service", "duration_ms": 150.0, "status_code": 200}}\n\n',
                # Event that doesn't match model filter
                'data: {"type": "request_complete", "data": {"model": "claude-3-5-haiku-20241022", "service_type": "proxy_service", "duration_ms": 150.0, "status_code": 200}}\n\n',
                # Event that doesn't match service type filter
                'data: {"type": "request_complete", "data": {"model": "claude-3-5-sonnet-20241022", "service_type": "sdk_service", "duration_ms": 150.0, "status_code": 200}}\n\n',
                # Event that doesn't match duration filter
                'data: {"type": "request_complete", "data": {"model": "claude-3-5-sonnet-20241022", "service_type": "proxy_service", "duration_ms": 50.0, "status_code": 200}}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "service_type": "proxy_service",
            "min_duration_ms": 100.0,
        }

        with client_no_auth.stream(
            "GET", "/logs/stream", params=cast(QueryParamTypes, params)
        ) as response:
            assert response.status_code == 200

            received_events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    received_events.append(event_data)
                    if len(received_events) >= 2:  # Connection + one filtered event
                        break

            # Should have connection event and one matching request_complete event
            assert len(received_events) == 2
            assert received_events[0]["type"] == "connection"
            assert received_events[1]["type"] == "request_complete"
            assert received_events[1]["data"]["model"] == "claude-3-5-sonnet-20241022"
            assert received_events[1]["data"]["service_type"] == "proxy_service"
            assert received_events[1]["data"]["duration_ms"] == 150.0

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_request_start_events_filtered(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test that filters are applied to request_start events."""

        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock events including request_start
        async def mock_events(connection_id=None, request_id=None):
            events = [
                # Connection event
                'data: {"type": "connection", "message": "Connected"}\n\n',
                # request_start that matches filters
                'data: {"type": "request_start", "data": {"model": "claude-3-5-sonnet-20241022", "service_type": "proxy_service"}}\n\n',
                # request_start that doesn't match
                'data: {"type": "request_start", "data": {"model": "claude-3-5-haiku-20241022", "service_type": "proxy_service"}}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"model": "claude-3-5-sonnet-20241022"}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            received_events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    received_events.append(event_data)
                    if len(received_events) >= 2:  # Connection + one filtered event
                        break

            # Should have connection event and one matching request_start event
            assert len(received_events) == 2
            assert received_events[0]["type"] == "connection"
            assert received_events[1]["type"] == "request_start"
            assert received_events[1]["data"]["model"] == "claude-3-5-sonnet-20241022"

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_system_events_not_filtered(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test that system events (connection, error, etc.) are not filtered."""

        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock system events that should always pass through
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
                'data: {"type": "error", "message": "Test error"}\n\n',
                'data: {"type": "overflow", "message": "Queue overflow"}\n\n',
                'data: {"type": "disconnect", "message": "Disconnected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        # Apply strict filters
        params = {
            "model": "claude-3-5-sonnet-20241022",
            "service_type": "proxy_service",
            "min_duration_ms": 1000.0,  # Very high filter
        }

        with client_no_auth.stream(
            "GET", "/logs/stream", params=cast(QueryParamTypes, params)
        ) as response:
            assert response.status_code == 200

            received_events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    event_data = json.loads(line[6:])
                    received_events.append(event_data)
                    if len(received_events) >= 4:  # All system events
                        break

            # All system events should pass through despite filters
            assert len(received_events) == 4
            assert received_events[0]["type"] == "connection"
            assert received_events[1]["type"] == "error"
            assert received_events[2]["type"] == "overflow"
            assert received_events[3]["type"] == "disconnect"

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_malformed_json_handled(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test that malformed JSON events are passed through."""

        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock events with malformed JSON
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
                "data: {invalid json}\n\n",  # Malformed JSON - should pass through
                'data: {"type": "request_complete", "data": {"model": "claude-3-5-sonnet-20241022"}}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {"model": "claude-3-5-sonnet-20241022"}

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            received_lines = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    received_lines.append(line)
                    if len(received_lines) >= 3:
                        break

            # All events should pass through (malformed JSON passes through)
            assert len(received_lines) == 3
            assert "Connected" in received_lines[0]
            assert (
                "{invalid json}" in received_lines[1]
            )  # Malformed JSON passed through
            assert "claude-3-5-sonnet-20241022" in received_lines[2]


class TestSSEStreamFilteringEdgeCases:
    """Test edge cases for SSE stream filtering."""

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_empty_filter_values(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with empty string filter values."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {
            "model": "",
            "service_type": "",
        }

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should still work (empty filters ignored)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break

    def test_sse_stream_invalid_numeric_filters(
        self, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with invalid numeric filter values."""
        # FastAPI should handle validation, but test the endpoint
        params = {
            "min_duration_ms": "invalid",
            "status_code_min": "not_a_number",
        }

        # This should result in a 422 validation error from FastAPI
        response = client_no_auth.get("/logs/stream", params=params)
        assert response.status_code == 422

    @patch("ccproxy.observability.sse_events.get_sse_manager")
    def test_sse_stream_negative_numeric_filters(
        self, mock_get_manager: AsyncMock, client_no_auth: TestClient
    ) -> None:
        """Test SSE stream with negative numeric filter values."""
        # Create mock SSE manager
        mock_manager = AsyncMock()
        mock_get_manager.return_value = mock_manager

        # Mock basic connection event
        async def mock_events(connection_id=None, request_id=None):
            events = [
                'data: {"type": "connection", "message": "Connected"}\n\n',
            ]
            for event in events:
                yield event

        mock_manager.add_connection = mock_events

        params = {
            "min_duration_ms": -100.0,
            "max_duration_ms": -50.0,
            "status_code_min": -1,
            "status_code_max": -1,
        }

        with client_no_auth.stream("GET", "/logs/stream", params=params) as response:
            assert response.status_code == 200

            # Should still connect (negative filters are valid but unlikely to match)
            for line in response.iter_lines():
                if line.startswith("data: "):
                    connection_data = json.loads(line[6:])
                    assert connection_data["type"] == "connection"
                    break
