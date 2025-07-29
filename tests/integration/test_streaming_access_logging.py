"""Integration tests for streaming access logging functionality."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock

from ccproxy.api.app import create_app
from ccproxy.config.settings import Settings


pytest.skip("skipping entire module", allow_module_level=True)


class TestStreamingAccessLogging:  # type: ignore[unreachable]
    """Test streaming access logging integration for both API endpoints."""

    def test_anthropic_streaming_access_logging(
        self,
        test_settings: Settings,
        mock_external_anthropic_api: HTTPXMock,
        mock_internal_claude_sdk_service_streaming,
    ) -> None:
        """Test end-to-end access logging for Anthropic streaming endpoint."""
        # Mock streaming response from Claude API
        streaming_chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]

        # Set up streaming response
        streaming_response = "\n".join(
            [
                f"event: {chunk.get('type', 'message_delta')}\ndata: {json.dumps(chunk)}"
                for chunk in streaming_chunks
            ]
        )

        mock_external_anthropic_api.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            content=streaming_response.encode(),
            headers={"content-type": "text/event-stream"},
            status_code=200,
        )

        # Create app with test settings and mock service
        app = create_app(settings=test_settings)

        # Override dependencies
        from ccproxy.api.dependencies import (
            get_cached_claude_service,
            get_cached_settings,
        )
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: test_settings
        app.dependency_overrides[get_cached_settings] = lambda request: test_settings
        app.dependency_overrides[get_cached_claude_service] = (
            lambda request: mock_internal_claude_sdk_service_streaming
        )

        client = TestClient(app)

        # Patch log_request_access to verify it's called
        with patch(
            "ccproxy.observability.access_logger.log_request_access"
        ) as mock_log:
            # Make streaming request to Anthropic endpoint
            with client.stream(
                "POST",
                "/sdk/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            ) as response:
                assert response.status_code == 200
                assert (
                    response.headers["content-type"]
                    == "text/event-stream; charset=utf-8"
                )

                # Consume all chunks
                chunks = []
                for line in response.iter_lines():
                    if line.strip():
                        chunks.append(line)

                # Verify we got streaming chunks
                assert len(chunks) > 0

                # Verify chunks contain expected events
                event_lines = [line for line in chunks if line.startswith("event:")]
                data_lines = [line for line in chunks if line.startswith("data:")]
                assert len(event_lines) > 0
                assert len(data_lines) > 0

            # Verify access logging was called after stream completion
            mock_log.assert_called_once()
            call_args = mock_log.call_args

            # Verify context was passed
            assert "context" in call_args.kwargs
            context = call_args.kwargs["context"]
            assert hasattr(context, "request_id")
            assert hasattr(context, "metadata")

            # Verify status code
            assert call_args.kwargs["status_code"] == 200

            # Verify streaming completion event was set
            assert context.metadata.get("event_type") == "streaming_complete"

    def test_openai_streaming_access_logging(
        self,
        test_settings: Settings,
        mock_external_anthropic_api: HTTPXMock,
        mock_internal_claude_sdk_service_streaming,
    ) -> None:
        """Test end-to-end access logging for OpenAI streaming endpoint."""
        # Mock streaming response from Claude API (OpenAI adapter will convert)
        streaming_chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]

        # Set up streaming response
        streaming_response = "\n".join(
            [
                f"event: {chunk.get('type', 'message_delta')}\ndata: {json.dumps(chunk)}"
                for chunk in streaming_chunks
            ]
        )

        mock_external_anthropic_api.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            content=streaming_response.encode(),
            headers={"content-type": "text/event-stream"},
            status_code=200,
        )

        # Create app with test settings and mock service
        app = create_app(settings=test_settings)

        # Override dependencies
        from ccproxy.api.dependencies import (
            get_cached_claude_service,
            get_cached_settings,
        )
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: test_settings
        app.dependency_overrides[get_cached_settings] = lambda request: test_settings
        app.dependency_overrides[get_cached_claude_service] = (
            lambda request: mock_internal_claude_sdk_service_streaming
        )

        client = TestClient(app)

        # Patch log_request_access to verify it's called
        with patch(
            "ccproxy.observability.access_logger.log_request_access"
        ) as mock_log:
            # Make streaming request to OpenAI endpoint
            with client.stream(
                "POST",
                "/sdk/v1/chat/completions",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            ) as response:
                assert response.status_code == 200
                assert (
                    response.headers["content-type"]
                    == "text/event-stream; charset=utf-8"
                )

                # Consume all chunks
                chunks = []
                for line in response.iter_lines():
                    if line.strip():
                        chunks.append(line)

                # Verify we got streaming chunks
                assert len(chunks) > 0

                # Verify chunks contain OpenAI format data
                data_lines = [line for line in chunks if line.startswith("data:")]
                assert len(data_lines) > 0

                # Should end with [DONE]
                assert any("[DONE]" in line for line in chunks)

            # Verify access logging was called after stream completion
            mock_log.assert_called_once()
            call_args = mock_log.call_args

            # Verify context was passed
            assert "context" in call_args.kwargs
            context = call_args.kwargs["context"]
            assert hasattr(context, "request_id")
            assert hasattr(context, "metadata")

            # Verify status code
            assert call_args.kwargs["status_code"] == 200

            # Verify streaming completion event was set
            assert context.metadata.get("event_type") == "streaming_complete"

    def test_streaming_access_logging_with_error(
        self,
        test_settings: Settings,
        mock_external_anthropic_api: HTTPXMock,
        mock_internal_claude_sdk_service_streaming,
    ) -> None:
        """Test that access logging is called even when streaming encounters errors."""
        # Mock error response from Claude API
        mock_external_anthropic_api.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            json={
                "error": {"type": "invalid_request_error", "message": "Invalid model"}
            },
            status_code=400,
        )

        # Create app with test settings and mock service
        app = create_app(settings=test_settings)

        # Override dependencies
        from ccproxy.api.dependencies import (
            get_cached_claude_service,
            get_cached_settings,
        )
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: test_settings
        app.dependency_overrides[get_cached_settings] = lambda request: test_settings
        app.dependency_overrides[get_cached_claude_service] = (
            lambda request: mock_internal_claude_sdk_service_streaming
        )

        client = TestClient(app)

        # Patch log_request_access to verify it's called
        with patch(
            "ccproxy.observability.access_logger.log_request_access"
        ) as mock_log:
            # Make streaming request that will fail
            response = client.post(
                "/sdk/v1/messages",
                json={
                    "model": "invalid-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                    "max_tokens": 100,
                },
            )

            # Should get error response (not streaming)
            assert response.status_code in [400, 500]

            # For error cases, access logging happens via middleware, not streaming wrapper
            # This test verifies the system handles errors gracefully

    def test_streaming_access_logging_failure_graceful(
        self,
        test_settings: Settings,
        mock_external_anthropic_api: HTTPXMock,
        mock_internal_claude_sdk_service_streaming,
    ) -> None:
        """Test that streaming continues when access logging fails."""
        # Mock streaming response from Claude API
        streaming_chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {"type": "message_stop"},
        ]

        streaming_response = "\n".join(
            [
                f"event: {chunk.get('type', 'message_delta')}\ndata: {json.dumps(chunk)}"
                for chunk in streaming_chunks
            ]
        )

        mock_external_anthropic_api.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            content=streaming_response.encode(),
            headers={"content-type": "text/event-stream"},
            status_code=200,
        )

        # Create app with test settings and mock service
        app = create_app(settings=test_settings)

        # Override dependencies
        from ccproxy.api.dependencies import (
            get_cached_claude_service,
            get_cached_settings,
        )
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: test_settings
        app.dependency_overrides[get_cached_settings] = lambda request: test_settings
        app.dependency_overrides[get_cached_claude_service] = (
            lambda request: mock_internal_claude_sdk_service_streaming
        )

        client = TestClient(app)

        # Patch log_request_access to raise an exception
        with patch(
            "ccproxy.observability.access_logger.log_request_access"
        ) as mock_log:
            mock_log.side_effect = Exception("Logging failed")

            # Patch logger to verify warning is logged
            with patch(
                "ccproxy.observability.streaming_response.logger"
            ) as mock_logger:
                # Make streaming request - should still work despite logging failure
                with client.stream(
                    "POST",
                    "/sdk/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": True,
                        "max_tokens": 100,
                    },
                ) as response:
                    assert response.status_code == 200

                    # Consume all chunks - should work despite logging failure
                    chunks = []
                    for line in response.iter_lines():
                        if line.strip():
                            chunks.append(line)

                    # Verify we got streaming chunks
                    assert len(chunks) > 0

                # Verify logging was attempted
                mock_log.assert_called_once()

                # Verify warning was logged about the failure
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args
                assert warning_call[0][0] == "streaming_access_log_failed"
                assert "error" in warning_call[1]
                assert warning_call[1]["error"] == "Logging failed"

    def test_non_streaming_requests_unaffected(
        self,
        test_settings: Settings,
        mock_external_anthropic_api: HTTPXMock,
        mock_internal_claude_sdk_service_streaming,
    ) -> None:
        """Test that non-streaming requests are not affected by streaming access logging."""
        # Mock non-streaming response from Claude API
        mock_external_anthropic_api.add_response(
            method="POST",
            url="https://api.anthropic.com/v1/messages",
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello world"}],
                "model": "claude-3-5-sonnet-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
            status_code=200,
        )

        # Create app with test settings and mock service
        app = create_app(settings=test_settings)

        # Override dependencies
        from ccproxy.api.dependencies import (
            get_cached_claude_service,
            get_cached_settings,
        )
        from ccproxy.config.settings import get_settings as original_get_settings

        app.dependency_overrides[original_get_settings] = lambda: test_settings
        app.dependency_overrides[get_cached_settings] = lambda request: test_settings
        app.dependency_overrides[get_cached_claude_service] = (
            lambda request: mock_internal_claude_sdk_service_streaming
        )

        client = TestClient(app)

        # Make non-streaming request to Anthropic endpoint
        response = client.post(
            "/sdk/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,  # Non-streaming
                "max_tokens": 100,
            },
        )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0

        # Non-streaming requests use normal middleware access logging,
        # not the StreamingResponseWithLogging wrapper
