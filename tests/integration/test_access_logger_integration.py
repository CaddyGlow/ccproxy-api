"""
Integration tests for access logger with queue-based DuckDB storage.

This module tests the integration between the access logger and
the queue-based storage solution to ensure end-to-end functionality.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlmodel import Session, select

from ccproxy.observability.access_logger import log_request_access
from ccproxy.observability.context import RequestContext
from ccproxy.observability.storage.duckdb_simple import SimpleDuckDBStorage
from ccproxy.observability.storage.models import AccessLog


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database path for testing."""
    return tmp_path / "test_access_logs.duckdb"


@pytest.fixture
async def storage_with_db(
    temp_db_path: Path,
) -> AsyncGenerator[SimpleDuckDBStorage, None]:
    """Create and initialize DuckDB storage for testing."""
    storage = SimpleDuckDBStorage(temp_db_path)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def mock_request_context() -> RequestContext:
    """Create mock request context for testing."""
    from tests.conftest import create_test_request_context

    context = create_test_request_context(
        request_id="test-context-123",
        method="POST",
        path="/v1/messages",
        endpoint="messages",
        model="claude-3-5-sonnet-20241022",
        streaming=False,
        service_type="proxy_service",
        status_code=200,
        tokens_input=100,
        tokens_output=50,
        cost_usd=0.002,
    )
    return context


class TestAccessLoggerIntegration:
    """Integration tests for access logger with storage."""

    async def test_log_request_access_stores_to_queue(
        self, storage_with_db: SimpleDuckDBStorage, mock_request_context: RequestContext
    ) -> None:
        """Test that log_request_access properly stores data via queue."""
        # Log access with storage
        await log_request_access(
            context=mock_request_context,
            status_code=200,
            client_ip="192.168.1.100",
            user_agent="test-client/1.0",
            method="POST",
            path="/v1/messages",
            query="stream=false",
            storage=storage_with_db,
        )

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify data was stored in database
        with Session(storage_with_db._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "test-context-123")
            ).first()

            assert result is not None
            assert result.request_id == "test-context-123"
            assert result.method == "POST"
            assert result.path == "/v1/messages"
            assert result.client_ip == "192.168.1.100"
            assert result.user_agent == "test-client/1.0"
            assert result.query == "stream=false"
            assert result.model == "claude-3-5-sonnet-20241022"
            assert result.tokens_input == 100
            assert result.tokens_output == 50
            assert result.cost_usd == pytest.approx(0.002)

    async def test_log_request_access_without_storage(
        self, mock_request_context: RequestContext
    ) -> None:
        """Test that log_request_access works without storage (no errors)."""
        # Should not raise any exceptions when storage is None
        await log_request_access(
            context=mock_request_context,
            status_code=200,
            client_ip="192.168.1.100",
            user_agent="test-client/1.0",
            storage=None,
        )

    async def test_multiple_concurrent_access_logs(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test multiple concurrent access log calls don't cause deadlocks."""
        from tests.conftest import create_test_request_context

        contexts = []
        for i in range(10):
            context = create_test_request_context(
                request_id=f"concurrent-context-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="messages",
                model="claude-3-5-sonnet-20241022",
                status_code=200,
                tokens_input=50 + i,
                tokens_output=25 + i,
                cost_usd=0.001 * (i + 1),
            )
            contexts.append(context)

        # Submit all access logs concurrently
        start_time = time.time()
        tasks = [
            log_request_access(
                context=ctx,
                status_code=200,
                client_ip=f"192.168.1.{100 + i}",
                user_agent="concurrent-client/1.0",
                storage=storage_with_db,
            )
            for i, ctx in enumerate(contexts)
        ]

        await asyncio.gather(*tasks)
        end_time = time.time()

        # Should complete quickly (no deadlocks)
        assert end_time - start_time < 2.0, "Concurrent access logs took too long"

        # Give background worker time to process
        await asyncio.sleep(0.3)

        # Verify all data was stored
        with Session(storage_with_db._engine) as session:
            results = session.exec(select(AccessLog)).all()
            assert len(results) == 10, f"Expected 10 records, got {len(results)}"

            # Verify each record
            for i, result in enumerate(results):
                assert result.request_id == f"concurrent-context-{i}"
                assert result.client_ip == f"192.168.1.{100 + i}"

    async def test_access_logger_handles_storage_errors(
        self, storage_with_db: SimpleDuckDBStorage, mock_request_context: RequestContext
    ) -> None:
        """Test that access logger handles storage errors gracefully."""
        # Mock storage to fail
        with patch.object(
            storage_with_db, "store_request", side_effect=Exception("Storage error")
        ):
            # Should not raise exception even if storage fails
            await log_request_access(
                context=mock_request_context,
                status_code=200,
                client_ip="192.168.1.100",
                user_agent="test-client/1.0",
                storage=storage_with_db,
            )

    async def test_streaming_access_log_integration(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test access logging for streaming requests."""
        from tests.conftest import create_test_request_context

        # Create streaming context
        context = create_test_request_context(
            request_id="streaming-test-123",
            method="POST",
            path="/v1/messages",
            endpoint="messages",
            model="claude-3-5-sonnet-20241022",
            streaming=True,
            service_type="proxy_service",
            status_code=200,
            tokens_input=150,
            tokens_output=75,
            cost_usd=0.003,
        )

        # Log streaming access
        await log_request_access(
            context=context,
            status_code=200,
            client_ip="10.0.0.1",
            user_agent="streaming-client/2.0",
            method="POST",
            path="/v1/messages",
            query="stream=true",
            storage=storage_with_db,
        )

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify streaming data was stored correctly
        with Session(storage_with_db._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "streaming-test-123")
            ).first()

            assert result is not None
            assert result.streaming is True
            assert result.query == "stream=true"
            assert result.tokens_input == 150
            assert result.tokens_output == 75

    async def test_access_logger_with_partial_data(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test access logger with minimal/partial data."""
        from tests.conftest import create_test_request_context

        # Create minimal context
        context = create_test_request_context(
            request_id="minimal-context-123",
            method="GET",
            path="/api/models",
            endpoint="models",
        )

        # Log with minimal data
        await log_request_access(
            context=context,
            status_code=200,
            storage=storage_with_db,
        )

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify data was stored with defaults
        with Session(storage_with_db._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "minimal-context-123")
            ).first()

            assert result is not None
            assert result.method == "GET"
            assert result.path == "/api/models"
            assert result.status_code == 200
            assert result.tokens_input == 0  # Default value
            assert result.tokens_output == 0  # Default value
            assert result.cost_usd == 0.0  # Default value

    async def test_access_logger_metadata_extraction(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test that access logger correctly extracts metadata from context."""
        from tests.conftest import create_test_request_context

        # Create context with metadata
        context = create_test_request_context(
            request_id="metadata-test-123",
            method="POST",
            path="/v1/chat/completions",  # OpenAI format path
            endpoint="chat/completions",
            model="gpt-4",  # Different model
            streaming=False,
            service_type="openai_adapter",
            status_code=201,  # Non-200 status
            tokens_input=200,
            tokens_output=100,
            cache_read_tokens=50,
            cache_write_tokens=25,
            cost_usd=0.005,
            cost_sdk_usd=0.001,
        )

        # Log without explicitly passing some parameters (should use context metadata)
        await log_request_access(
            context=context,
            client_ip="203.0.113.1",
            user_agent="openai-client/1.0",
            storage=storage_with_db,
        )

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify metadata was correctly extracted and stored
        with Session(storage_with_db._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "metadata-test-123")
            ).first()

            assert result is not None
            assert result.method == "POST"  # From context metadata
            assert result.path == "/v1/chat/completions"  # From context metadata
            assert result.status_code == 201  # From context metadata
            assert result.model == "gpt-4"
            assert result.service_type == "openai_adapter"
            assert result.tokens_input == 200
            assert result.tokens_output == 100
            assert result.cache_read_tokens == 50
            assert result.cache_write_tokens == 25
            assert result.cost_usd == pytest.approx(0.005)
            assert result.cost_sdk_usd == pytest.approx(0.001)

    async def test_access_logger_error_with_message(
        self, storage_with_db: SimpleDuckDBStorage, mock_request_context: RequestContext
    ) -> None:
        """Test access logging with error message."""
        # Log access with error
        await log_request_access(
            context=mock_request_context,
            status_code=400,
            client_ip="192.168.1.100",
            user_agent="error-client/1.0",
            error_message="Invalid request format",
            storage=storage_with_db,
        )

        # Give background worker time to process
        await asyncio.sleep(0.2)

        # Verify error was logged (note: error_message is not stored in current schema)
        with Session(storage_with_db._engine) as session:
            result = session.exec(
                select(AccessLog).where(AccessLog.request_id == "test-context-123")
            ).first()

            assert result is not None
            assert result.status_code == 400
            # Note: error_message field doesn't exist in AccessLog model
            # This tests that the logger handles extra fields gracefully


class TestAccessLoggerPerformance:
    """Performance tests for access logger integration."""

    @pytest.mark.unit
    async def test_high_volume_access_logging(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test high-volume access logging performance."""
        from tests.conftest import create_test_request_context

        num_logs = 100
        contexts = []

        # Generate many contexts
        for i in range(num_logs):
            context = create_test_request_context(
                request_id=f"perf-test-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="messages",
                model="claude-3-5-sonnet-20241022",
                status_code=200,
                tokens_input=100,
                tokens_output=50,
                cost_usd=0.002,
            )
            contexts.append(context)

        # Log all access logs
        start_time = time.time()
        tasks = [
            log_request_access(
                context=ctx,
                status_code=200,
                client_ip=f"10.0.{i // 256}.{i % 256}",
                user_agent="perf-client/1.0",
                storage=storage_with_db,
            )
            for i, ctx in enumerate(contexts)
        ]

        await asyncio.gather(*tasks)
        log_time = time.time() - start_time

        # Should complete quickly
        assert log_time < 5.0, f"High-volume logging took too long: {log_time}s"

        # Give background worker time to process with retries
        for _attempt in range(10):
            await asyncio.sleep(0.5)
            with Session(storage_with_db._engine) as session:
                count = len(session.exec(select(AccessLog)).all())
                if count == num_logs:
                    break
        else:
            # Final check with detailed error
            with Session(storage_with_db._engine) as session:
                count = len(session.exec(select(AccessLog)).all())
                assert count == num_logs, (
                    f"Expected {num_logs} logs, got {count} after 5s wait"
                )

    @pytest.mark.unit
    async def test_mixed_streaming_and_regular_logs(
        self, storage_with_db: SimpleDuckDBStorage
    ) -> None:
        """Test mixed streaming and regular request logging."""
        tasks = []

        from tests.conftest import create_test_request_context

        # Create mix of streaming and regular requests
        for i in range(20):
            is_streaming = i % 2 == 0
            context = create_test_request_context(
                request_id=f"mixed-test-{i}",
                method="POST",
                path="/v1/messages",
                endpoint="messages",
                model="claude-3-5-sonnet-20241022",
                streaming=is_streaming,
                status_code=200,
                tokens_input=100 + i,
                tokens_output=50 + i,
                cost_usd=0.002 + (i * 0.001),
            )

            task = log_request_access(
                context=context,
                status_code=200,
                client_ip=f"172.16.{i // 256}.{i % 256}",
                user_agent="mixed-client/1.0",
                query=f"stream={str(is_streaming).lower()}",
                storage=storage_with_db,
            )
            tasks.append(task)

        # Execute all concurrently
        await asyncio.gather(*tasks)

        # Give background worker time to process with retries
        for _attempt in range(10):
            await asyncio.sleep(0.3)
            with Session(storage_with_db._engine) as session:
                results = session.exec(
                    select(AccessLog).order_by(AccessLog.request_id)
                ).all()
                if len(results) == 20:
                    break
        else:
            # Final check with detailed error
            with Session(storage_with_db._engine) as session:
                results = session.exec(
                    select(AccessLog).order_by(AccessLog.request_id)
                ).all()
                assert len(results) == 20, (
                    f"Expected 20 logs, got {len(results)} after 3s wait"
                )

            # Verify streaming flags are correct
            for i, result in enumerate(results):
                expected_streaming = i % 2 == 0
                assert result.streaming == expected_streaming
                assert result.query == f"stream={str(expected_streaming).lower()}"
