"""Basic coverage tests for service classes.

This module provides simple tests to improve coverage for the service classes
that had low test coverage. These tests focus on basic functionality that
can be easily tested without complex dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from ccproxy.metrics.collector import MetricsCollector
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage
from ccproxy.services.credentials.manager import CredentialsManager
from ccproxy.services.proxy_service import ProxyService


class TestServicesCoverage:
    """Tests to improve basic coverage for service classes."""
    
    @pytest.mark.unit
    def test_proxy_service_init(self, test_settings) -> None:
        """Test ProxyService basic initialization."""
        credentials_manager = CredentialsManager(test_settings.auth)
        storage = InMemoryMetricsStorage()
        metrics_collector = MetricsCollector(storage=storage)
        mock_proxy_client = Mock()
        
        proxy_service = ProxyService(
            proxy_client=mock_proxy_client,
            credentials_manager=credentials_manager,
            proxy_mode="direct",
            target_base_url="https://api.anthropic.com",
            metrics_collector=metrics_collector,
        )
        
        # Test basic properties
        assert proxy_service.proxy_mode == "direct"
        assert proxy_service.target_base_url == "https://api.anthropic.com"
        assert proxy_service.credentials_manager is not None
        assert proxy_service.metrics_collector is not None

    @pytest.mark.unit
    def test_proxy_service_should_stream_false(self, test_settings) -> None:
        """Test stream detection returns False correctly."""
        credentials_manager = CredentialsManager(test_settings.auth)
        storage = InMemoryMetricsStorage()
        metrics_collector = MetricsCollector(storage=storage)
        mock_proxy_client = Mock()
        
        proxy_service = ProxyService(
            proxy_client=mock_proxy_client,
            credentials_manager=credentials_manager,
            proxy_mode="direct",
            target_base_url="https://api.anthropic.com",
            metrics_collector=metrics_collector,
        )
        
        # Test with stream=False
        request_data = {"stream": False, "messages": []}
        assert not proxy_service._should_stream_response(request_data)
        
        # Test with missing stream key
        request_data = {"messages": []}
        assert not proxy_service._should_stream_response(request_data)

    @pytest.mark.unit
    def test_credentials_manager_init(self, test_settings) -> None:
        """Test CredentialsManager basic initialization."""
        manager = CredentialsManager(test_settings.auth)
        
        # Test basic properties
        assert manager.config is not None
        assert manager._storage is not None
        assert manager._oauth_client is not None
        assert manager._refresh_lock is not None

    @pytest.mark.unit
    def test_credentials_manager_storage_property(self, test_settings) -> None:
        """Test CredentialsManager storage property."""
        manager = CredentialsManager(test_settings.auth)
        
        # Test storage property
        storage = manager.storage
        assert storage is not None
        
        # Test that multiple calls return same instance
        assert manager.storage is storage

    @pytest.mark.unit
    def test_credentials_manager_determine_subscription_type(self, test_settings) -> None:
        """Test subscription type determination."""
        manager = CredentialsManager(test_settings.auth)
        
        # Test with profile containing subscription
        profile_with_sub = {
            "subscription": {"type": "pro"},
            "user_id": "user_123",
        }
        sub_type = manager._determine_subscription_type(profile_with_sub)
        assert sub_type == "pro"
        
        # Test with profile without subscription
        profile_without_sub = {"user_id": "user_123"}
        sub_type = manager._determine_subscription_type(profile_without_sub)
        assert sub_type == "free"

    @pytest.mark.unit
    def test_credentials_manager_should_refresh_token(self, test_settings) -> None:
        """Test token refresh condition logic."""
        from datetime import datetime, timedelta
        
        manager = CredentialsManager(test_settings.auth)
        
        # Test with token expiring soon (should refresh)
        expiring_credentials = {
            "access_token": "expiring_token",
            "refresh_token": "test_refresh",
            "expires_at": (datetime.now() + timedelta(minutes=1)).isoformat(),
            "user_id": "user_123",
        }
        assert manager._should_refresh_token(expiring_credentials)
        
        # Test with token valid for long time (should not refresh)
        valid_credentials = {
            "access_token": "valid_token", 
            "refresh_token": "test_refresh",
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            "user_id": "user_123",
        }
        assert not manager._should_refresh_token(valid_credentials)

    @pytest.mark.unit
    def test_credentials_manager_find_existing_path(self, test_settings, tmp_path) -> None:
        """Test finding existing storage path."""
        from pathlib import Path
        from ccproxy.services.credentials.config import CredentialsConfig
        
        # Create test paths
        path1 = tmp_path / ".claude1"
        path2 = tmp_path / ".claude2"
        path2.mkdir(parents=True, exist_ok=True)  # Only create path2
        
        config = CredentialsConfig(
            storage_paths=[path1, path2],
            oauth=test_settings.auth.oauth,
        )
        
        manager = CredentialsManager(config=config)
        found_path = manager._find_existing_path()
        assert found_path == path2  # Should find the existing path

    @pytest.mark.unit  
    def test_metrics_service_init(self) -> None:
        """Test MetricsService basic initialization."""
        from ccproxy.services.metrics_service import MetricsService
        
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage=storage)
        
        metrics_service = MetricsService(
            collector=collector,
            exporters=[],
            buffer_size=100,
            export_interval=60,
        )
        
        assert metrics_service.exporters == []
        assert metrics_service.buffer_size == 100
        assert metrics_service.export_interval == 60
        assert len(metrics_service._metrics_buffer) == 0

    @pytest.mark.unit
    def test_metrics_service_record_request_start(self) -> None:
        """Test recording request start metrics."""
        from ccproxy.services.metrics_service import MetricsService
        
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage=storage)
        
        metrics_service = MetricsService(
            collector=collector,
            exporters=[],
            buffer_size=100,
            export_interval=60,
        )
        
        request_id = "req_123"
        user_id = "user_456"
        model = "claude-3-5-sonnet-20241022"
        endpoint = "/v1/messages"
        
        metrics_service.record_request_start(
            request_id=request_id,
            user_id=user_id,
            model=model,
            endpoint=endpoint,
        )
        
        # Check that request metrics were updated
        assert request_id in metrics_service._request_metrics
        request_metric = metrics_service._request_metrics[request_id]
        assert request_metric["user_id"] == user_id
        assert request_metric["model"] == model
        assert request_metric["endpoint"] == endpoint

    @pytest.mark.unit
    def test_metrics_service_should_export(self) -> None:
        """Test export condition logic."""
        from ccproxy.services.metrics_service import MetricsService
        
        storage = InMemoryMetricsStorage()
        collector = MetricsCollector(storage=storage)
        
        # Test with small buffer (should export when full)
        metrics_service = MetricsService(
            collector=collector,
            exporters=[],
            buffer_size=2,  # Small buffer for testing
            export_interval=60,
        )
        
        # Fill the buffer
        for i in range(3):  # More than buffer size
            request_id = f"req_{i}"
            metrics_service.record_request_start(
                request_id=request_id,
                user_id="user_456",
                model="claude-3-5-sonnet-20241022",
                endpoint="/v1/messages",
            )
            metrics_service.record_request_end(
                request_id=request_id,
                status_code=200,
                tokens_used={"input": 10, "output": 5},
                response_size=150,
            )
        
        assert metrics_service.should_export()

    @pytest.mark.unit
    def test_oauth_client_init(self, test_settings) -> None:
        """Test OAuthClient basic initialization."""
        from ccproxy.services.credentials.oauth_client import OAuthClient
        
        client = OAuthClient(config=test_settings.auth.oauth)
        
        # Test basic properties
        assert client.config is not None

    @pytest.mark.unit
    def test_oauth_client_generate_pkce_pair(self, test_settings) -> None:
        """Test PKCE code generation."""
        from ccproxy.services.credentials.oauth_client import OAuthClient
        
        client = OAuthClient(config=test_settings.auth.oauth)
        
        verifier, challenge = client.generate_pkce_pair()
        
        # Verify verifier format
        assert len(verifier) >= 43  # PKCE spec minimum
        assert len(verifier) <= 128  # PKCE spec maximum
        
        # Verify challenge format
        assert len(challenge) == 43  # SHA256 base64url length without padding

    @pytest.mark.unit
    def test_oauth_client_pkce_uniqueness(self, test_settings) -> None:
        """Test that PKCE pairs are unique."""
        from ccproxy.services.credentials.oauth_client import OAuthClient
        
        client = OAuthClient(config=test_settings.auth.oauth)
        
        # Generate multiple pairs
        pairs = [client.generate_pkce_pair() for _ in range(5)]
        verifiers = [pair[0] for pair in pairs]
        challenges = [pair[1] for pair in pairs]
        
        # All should be unique
        assert len(set(verifiers)) == len(verifiers)
        assert len(set(challenges)) == len(challenges)