"""Integration tests for Claude instance connection pool."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_code_proxy.config.settings import PoolSettings, Settings
from claude_code_proxy.services.pool_manager import pool_manager


@pytest.mark.integration
class TestPoolIntegration:
    """Integration tests for connection pool with API endpoints."""

    @pytest.fixture
    def mock_settings_with_pool(self):
        """Create settings with pool enabled."""
        settings = Settings()
        settings.pool_settings = PoolSettings(
            enabled=True,
            min_size=2,
            max_size=5,
            idle_timeout=300,
            warmup_on_startup=True,
        )
        return settings

    @pytest.fixture
    def mock_settings_no_pool(self):
        """Create settings with pool disabled."""
        settings = Settings()
        settings.pool_settings = PoolSettings(enabled=False)
        return settings

    @pytest.mark.asyncio
    async def test_pool_manager_configuration(self, mock_settings_with_pool):
        """Test pool manager configures correctly."""
        pool_manager.configure(mock_settings_with_pool)

        assert pool_manager.is_enabled
        assert pool_manager._pool is not None
        assert pool_manager._pool.min_size == 2
        assert pool_manager._pool.max_size == 5

    @pytest.mark.asyncio
    async def test_pool_manager_disabled(self, mock_settings_no_pool):
        """Test pool manager when pooling is disabled."""
        pool_manager.configure(mock_settings_no_pool)

        assert not pool_manager.is_enabled
        assert pool_manager._pool is None

        # Should still work, creating new clients
        client, conn = await pool_manager.acquire_client()
        assert client is not None
        assert conn is None  # No pooled connection when disabled

    @pytest.mark.asyncio
    async def test_pool_warmup_on_startup(self, mock_settings_with_pool):
        """Test pool warmup during initialization."""
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()

            await pool_manager.initialize()

            stats = pool_manager.get_stats()
            assert (
                stats["connections_created"]
                >= mock_settings_with_pool.pool_settings.min_size
            )

    @pytest.mark.asyncio
    async def test_concurrent_requests_use_pool(self, mock_settings_with_pool):
        """Test multiple concurrent requests efficiently use the pool."""
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()
            await pool_manager.initialize()

            # Simulate concurrent requests
            async def make_request():
                client, conn = await pool_manager.acquire_client()
                await asyncio.sleep(0.1)  # Simulate work
                await pool_manager.release_client(conn)
                return conn.id if conn else None

            # Run 10 concurrent requests
            tasks = [make_request() for _ in range(10)]
            results = await asyncio.gather(*tasks)

            # Should reuse connections (fewer unique IDs than requests)
            unique_ids: set[str] = set(filter(None, results))
            assert len(unique_ids) <= mock_settings_with_pool.pool_settings.max_size
            assert len(unique_ids) < 10  # Some reuse occurred

    @pytest.mark.asyncio
    async def test_api_endpoint_with_pool(self, test_app, mock_settings_with_pool):
        """Test API endpoint uses pool correctly."""
        with patch(
            "claude_code_proxy.config.settings.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_settings_with_pool
            pool_manager.configure(mock_settings_with_pool)

            with patch(
                "claude_code_proxy.services.claude_pool.ClaudeClient"
            ) as mock_client:
                # Mock the Claude client
                mock_instance = MagicMock()
                mock_instance.create_completion = AsyncMock(
                    return_value={
                        "id": "test_id",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Test response"}],
                        "model": "claude-3-opus-20240229",
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                )
                mock_client.return_value = mock_instance

                # Initialize pool
                await pool_manager.initialize()

                # Make request
                client = TestClient(test_app)
                response = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3-opus-20240229",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 100,
                    },
                )

                assert response.status_code == 200
                assert response.json()["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_performance_improvement_with_pool(self, mock_settings_with_pool):
        """Test that pooling improves performance."""
        # Test with pool
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()
            await pool_manager.initialize()

            # Measure time for pooled requests
            start_time = time.time()
            for _ in range(5):
                client, conn = await pool_manager.acquire_client()
                await pool_manager.release_client(conn)
            pooled_time = time.time() - start_time

            # Cleanup
            await pool_manager.shutdown()

            # Test without pool (simulate creation overhead)
            mock_settings_no_pool = Settings()
            mock_settings_no_pool.pool_settings = PoolSettings(enabled=False)
            pool_manager.configure(mock_settings_no_pool)

            # Add artificial delay to simulate creation overhead
            def slow_init(*args, **kwargs):
                time.sleep(0.05)  # 50ms creation time
                return MagicMock()

            mock_client.side_effect = slow_init

            start_time = time.time()
            for _ in range(5):
                client, conn = await pool_manager.acquire_client()
                # No release needed when pool is disabled
            non_pooled_time = time.time() - start_time

            # Pooled should be significantly faster
            assert pooled_time < non_pooled_time * 0.5  # At least 2x faster

    @pytest.mark.asyncio
    async def test_pool_error_handling(self, mock_settings_with_pool):
        """Test pool handles errors gracefully."""
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            # Make client creation fail sometimes
            call_count = 0

            def maybe_fail(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count % 3 == 0:
                    raise Exception("Simulated failure")
                return MagicMock()

            mock_client.side_effect = maybe_fail

            # Pool should handle failures and continue
            await pool_manager.initialize()

            # Should still be able to acquire some connections
            successful_acquires = 0
            for _ in range(10):
                try:
                    client, conn = await pool_manager.acquire_client()
                    if conn:
                        successful_acquires += 1
                        await pool_manager.release_client(conn)
                except Exception:
                    pass

            assert successful_acquires > 0

    @pytest.mark.asyncio
    async def test_pool_cleanup_on_shutdown(self, mock_settings_with_pool):
        """Test pool cleans up properly on shutdown."""
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()
            await pool_manager.initialize()

            # Acquire some connections
            connections = []
            for _ in range(3):
                client, conn = await pool_manager.acquire_client()
                connections.append(conn)

            # Shutdown should clean everything
            await pool_manager.shutdown()

            assert not pool_manager.is_enabled
            assert pool_manager._pool is None


@pytest.mark.integration
class TestPoolWithStreaming:
    """Test pool integration with streaming responses."""

    @pytest.mark.asyncio
    async def test_streaming_with_pool(self, test_app):
        """Test streaming responses work correctly with pooling."""
        settings = Settings()
        settings.pool_settings = PoolSettings(enabled=True, min_size=1, max_size=3)

        with patch(
            "claude_code_proxy.config.settings.get_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = settings
            pool_manager.configure(settings)

            with patch(
                "claude_code_proxy.services.claude_pool.ClaudeClient"
            ) as mock_client:
                # Mock streaming response
                async def mock_stream(*args, **kwargs):
                    yield {
                        "type": "message_start",
                        "message": {"id": "test", "type": "message"},
                    }
                    yield {"type": "content_block_delta", "delta": {"text": "Hello"}}
                    yield {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                    }

                mock_instance = MagicMock()
                mock_instance.create_completion = AsyncMock(return_value=mock_stream())
                mock_client.return_value = mock_instance

                await pool_manager.initialize()

                client = TestClient(test_app)
                with client.stream(
                    "POST",
                    "/v1/messages",
                    json={
                        "model": "claude-3-opus-20240229",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 100,
                        "stream": True,
                    },
                ) as response:
                    assert response.status_code == 200
                    chunks = list(response.iter_lines())
                    assert len(chunks) > 0


@pytest.mark.integration
class TestPoolMetrics:
    """Test pool metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_pool_statistics(self, mock_settings_with_pool):
        """Test pool provides accurate statistics."""
        pool_manager.configure(mock_settings_with_pool)

        with patch(
            "claude_code_proxy.services.claude_pool.ClaudeClient"
        ) as mock_client:
            mock_client.return_value = MagicMock()
            await pool_manager.initialize()

            initial_stats = pool_manager.get_stats()
            assert (
                initial_stats["connections_created"]
                >= mock_settings_with_pool.pool_settings.min_size
            )

            # Acquire and release connections
            for _ in range(5):
                client, conn = await pool_manager.acquire_client()
                await pool_manager.release_client(conn)

            final_stats = pool_manager.get_stats()
            assert final_stats["connections_reused"] >= 5
            assert (
                final_stats["total_connections"]
                <= mock_settings_with_pool.pool_settings.max_size
            )
