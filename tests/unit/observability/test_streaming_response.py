"""Unit tests for StreamingResponseWithLogging utility class."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.observability.context import RequestContext
from ccproxy.observability.streaming_response import StreamingResponseWithLogging


class TestStreamingResponseWithLogging:
    """Test StreamingResponseWithLogging functionality."""

    @pytest.fixture
    def mock_request_context(self) -> MagicMock:
        """Create a mock request context for testing."""
        context = MagicMock(spec=RequestContext)
        context.request_id = "test-request-123"
        context.add_metadata = MagicMock()
        return context

    @pytest.fixture
    def mock_metrics(self) -> MagicMock:
        """Create a mock PrometheusMetrics instance."""
        return MagicMock()

    async def sample_content_generator(self) -> AsyncGenerator[bytes, None]:
        """Sample content generator for testing."""
        yield b"data: chunk1\n\n"
        yield b"data: chunk2\n\n"
        yield b"data: [DONE]\n\n"

    async def failing_content_generator(self) -> AsyncGenerator[bytes, None]:
        """Content generator that raises an exception."""
        yield b"data: chunk1\n\n"
        raise ValueError("Test error in generator")

    @pytest.mark.asyncio
    async def test_streaming_response_logs_on_completion(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test that access logging is triggered when stream completes successfully."""
        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            mock_log.return_value = AsyncMock()

            # Create streaming response
            response = StreamingResponseWithLogging(
                content=self.sample_content_generator(),
                request_context=mock_request_context,
                metrics=mock_metrics,
                status_code=200,
                media_type="text/event-stream",
            )

            # Consume all content from the stream
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            # Verify we got the expected chunks
            assert len(chunks) == 3
            assert chunks[0] == b"data: chunk1\n\n"
            assert chunks[1] == b"data: chunk2\n\n"
            assert chunks[2] == b"data: [DONE]\n\n"

            # Verify access logging was called
            mock_log.assert_called_once_with(
                context=mock_request_context,
                status_code=200,
                metrics=mock_metrics,
            )

            # Verify context metadata was updated with streaming completion event
            mock_request_context.add_metadata.assert_called_once_with(
                event_type="streaming_complete"
            )

    @pytest.mark.asyncio
    async def test_streaming_response_logs_on_error(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test that access logging is triggered even when stream fails."""
        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            mock_log.return_value = AsyncMock()

            # Create streaming response with failing generator
            response = StreamingResponseWithLogging(
                content=self.failing_content_generator(),
                request_context=mock_request_context,
                metrics=mock_metrics,
                status_code=200,
                media_type="text/event-stream",
            )

            # Try to consume content - should raise ValueError
            with pytest.raises(ValueError, match="Test error in generator"):
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)

            # Verify access logging was still called despite the error
            mock_log.assert_called_once_with(
                context=mock_request_context,
                status_code=200,
                metrics=mock_metrics,
            )

            # Verify context metadata was updated with streaming completion event
            mock_request_context.add_metadata.assert_called_once_with(
                event_type="streaming_complete"
            )

    @pytest.mark.asyncio
    async def test_streaming_response_handles_logging_errors(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test graceful handling when access logging itself fails."""
        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            # Make log_request_access raise an exception
            mock_log.side_effect = Exception("Logging failed")

            with patch(
                "ccproxy.observability.streaming_response.logger"
            ) as mock_logger:
                # Create streaming response
                response = StreamingResponseWithLogging(
                    content=self.sample_content_generator(),
                    request_context=mock_request_context,
                    metrics=mock_metrics,
                    status_code=200,
                    media_type="text/event-stream",
                )

                # Consume all content - should not raise despite logging error
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk)

                # Verify we got the expected chunks despite logging failure
                assert len(chunks) == 3

                # Verify warning was logged about the failure
                mock_logger.warning.assert_called_once_with(
                    "streaming_access_log_failed",
                    error="Logging failed",
                    request_id="test-request-123",
                )

    @pytest.mark.asyncio
    async def test_streaming_response_without_metrics(
        self, mock_request_context: MagicMock
    ) -> None:
        """Test streaming response works without metrics instance."""
        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            mock_log.return_value = AsyncMock()

            # Create streaming response without metrics
            response = StreamingResponseWithLogging(
                content=self.sample_content_generator(),
                request_context=mock_request_context,
                metrics=None,  # No metrics
                status_code=200,
                media_type="text/event-stream",
            )

            # Consume all content
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            # Verify access logging was called with None metrics
            mock_log.assert_called_once_with(
                context=mock_request_context,
                status_code=200,
                metrics=None,
            )

    @pytest.mark.asyncio
    async def test_streaming_response_custom_status_code(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test streaming response with custom status code."""
        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            mock_log.return_value = AsyncMock()

            # Create streaming response with custom status code
            response = StreamingResponseWithLogging(
                content=self.sample_content_generator(),
                request_context=mock_request_context,
                metrics=mock_metrics,
                status_code=201,  # Custom status code
                media_type="text/event-stream",
            )

            # Consume all content
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            # Verify access logging was called with correct status code
            mock_log.assert_called_once_with(
                context=mock_request_context,
                status_code=201,
                metrics=mock_metrics,
            )

    def test_streaming_response_initialization(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test StreamingResponseWithLogging initialization."""
        # Create streaming response
        response = StreamingResponseWithLogging(
            content=self.sample_content_generator(),
            request_context=mock_request_context,
            metrics=mock_metrics,
            status_code=200,
            media_type="text/event-stream",
            headers={"Custom-Header": "test-value"},
        )

        # Verify basic properties
        assert response.status_code == 200
        assert response.media_type == "text/event-stream"
        assert response.headers["Custom-Header"] == "test-value"

    @pytest.mark.asyncio
    async def test_empty_content_generator(
        self, mock_request_context: MagicMock, mock_metrics: MagicMock
    ) -> None:
        """Test streaming response with empty content generator."""

        async def empty_generator() -> AsyncGenerator[bytes, None]:
            """Empty generator that yields nothing."""
            # Make this a proper empty async generator by using an empty loop
            for _ in []:  # Empty list, so loop never executes
                yield b"never reached"

        with patch(
            "ccproxy.observability.streaming_response.log_request_access"
        ) as mock_log:
            mock_log.return_value = AsyncMock()

            # Create streaming response with empty generator
            response = StreamingResponseWithLogging(
                content=empty_generator(),
                request_context=mock_request_context,
                metrics=mock_metrics,
                status_code=200,
                media_type="text/event-stream",
            )

            # Consume all content (should be empty)
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            # Should have no chunks
            assert len(chunks) == 0

            # Access logging should still be called
            mock_log.assert_called_once_with(
                context=mock_request_context,
                status_code=200,
                metrics=mock_metrics,
            )
