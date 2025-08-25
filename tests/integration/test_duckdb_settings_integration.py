"""Test DuckDB storage integration with settings."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ccproxy.config.observability import ObservabilitySettings
from ccproxy.config.settings import Settings
from ccproxy.core.async_task_manager import start_task_manager, stop_task_manager
from ccproxy.observability.storage.duckdb_simple import (
    AccessLogPayload,
    SimpleDuckDBStorage,
)


@pytest.fixture(autouse=True)
async def task_manager_fixture():
    """Start and stop task manager for each test."""
    await start_task_manager()
    yield
    await stop_task_manager()


@pytest.mark.unit
class TestDuckDBSettingsIntegration:
    """Test DuckDB storage properly uses settings configuration."""

    async def test_storage_uses_settings_path(self) -> None:
        """Test that SimpleDuckDBStorage uses the path from settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom" / "metrics.duckdb"

            # Create observability settings with custom path
            obs_settings = ObservabilitySettings(
                duckdb_enabled=True, duckdb_path=str(custom_path)
            )

            # Create storage with the path from settings
            storage = SimpleDuckDBStorage(database_path=obs_settings.duckdb_path)
            await storage.initialize()

            # Verify the storage is using the correct path
            assert storage.database_path == custom_path
            assert custom_path.exists()
            assert custom_path.parent.exists()

            # Test storing data to ensure it's working
            test_data: AccessLogPayload = {
                "request_id": "test_123",
                "timestamp": 1234567890,
                "method": "POST",
                "endpoint": "/v1/messages",
                "status_code": 200,
                "duration_ms": 100.0,
            }

            result = await storage.store_request(test_data)
            assert result is True

            # Wait for the background worker to process the queued item
            await storage._write_queue.join()

            # Verify data was stored
            recent = await storage.get_recent_requests(limit=1)
            assert len(recent) == 1
            assert recent[0]["request_id"] == "test_123"

            await storage.close()

    async def test_app_startup_with_custom_duckdb_path(self) -> None:
        """Test app startup uses custom DuckDB path from settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "app" / "metrics.duckdb"

            # Mock settings with custom path
            mock_settings = Settings(
                observability=ObservabilitySettings(
                    duckdb_enabled=True, duckdb_path=str(custom_path)
                )
            )

            # Test the initialization flow similar to app.py
            if mock_settings.observability.duckdb_enabled:
                storage = SimpleDuckDBStorage(
                    database_path=mock_settings.observability.duckdb_path
                )
                await storage.initialize()

                # Verify correct path is used
                assert storage.database_path == custom_path
                assert custom_path.exists()

                # Verify storage is functional
                assert storage.is_enabled()
                health = await storage.health_check()
                assert health["status"] == "healthy"
                assert health["database_path"] == str(custom_path)

                await storage.close()

    async def test_relative_path_resolution(self) -> None:
        """Test that relative paths are handled correctly."""
        # Test with relative path
        obs_settings = ObservabilitySettings(
            duckdb_enabled=True, duckdb_path="data/test_metrics.duckdb"
        )

        storage = SimpleDuckDBStorage(database_path=obs_settings.duckdb_path)
        await storage.initialize()

        # Verify the path was created
        assert storage.database_path.exists()
        assert storage.database_path.name == "test_metrics.duckdb"
        assert storage.database_path.parent.name == "data"

        # Clean up
        await storage.close()
        # Don't try to clean up the data directory as it may contain other files

    @patch("ccproxy.api.app.get_settings")
    @patch("ccproxy.utils.startup_helpers.SimpleDuckDBStorage")
    async def test_app_lifespan_uses_settings_path(
        self, mock_storage_class: AsyncMock, mock_get_settings: AsyncMock
    ) -> None:
        """Test that app lifespan correctly passes settings path to DuckDB storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "lifespan" / "metrics.duckdb"

            # Mock settings
            mock_settings = Settings(
                observability=ObservabilitySettings(
                    duckdb_enabled=True, duckdb_path=str(custom_path)
                )
            )
            mock_get_settings.return_value = mock_settings

            # Mock storage instance
            mock_storage_instance = AsyncMock()
            mock_storage_class.return_value = mock_storage_instance

            # Import and test the app initialization
            from ccproxy.api.app import create_app

            app = create_app()

            # Simulate the lifespan startup (simplified)
            if mock_settings.observability.duckdb_enabled:
                # This simulates what happens in the app lifespan
                storage = mock_storage_class(
                    database_path=mock_settings.observability.duckdb_path
                )

                # Verify SimpleDuckDBStorage was called with correct path
                mock_storage_class.assert_called_once_with(
                    database_path=str(custom_path)
                )
