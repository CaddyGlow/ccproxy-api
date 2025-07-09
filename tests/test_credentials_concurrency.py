"""Tests for concurrent token refresh and atomic file operations."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.services.credentials import (
    CredentialsConfig,
    CredentialsManager,
    JsonFileStorage,
    OAuthClient,
)
from ccproxy.services.credentials.exceptions import OAuthTokenRefreshError
from ccproxy.services.credentials.models import ClaudeCredentials, OAuthToken


class TestConcurrentTokenRefresh:
    """Test concurrent token refresh scenarios."""

    @pytest.fixture
    def expired_token(self):
        """Create an expired token."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        return OAuthToken(
            accessToken="expired-token",
            refreshToken="test-refresh-token",
            expiresAt=past_ms,
            scopes=["user:inference"],
            subscriptionType="pro",
        )

    @pytest.fixture
    def valid_token(self):
        """Create a valid token."""
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        return OAuthToken(
            accessToken="valid-token",
            refreshToken="test-refresh-token",
            expiresAt=future_ms,
            scopes=["user:inference"],
            subscriptionType="pro",
        )

    @pytest.mark.asyncio
    async def test_concurrent_refresh_only_refreshes_once(
        self, tmp_path, expired_token, valid_token
    ):
        """Test that concurrent requests only trigger one refresh."""
        storage_path = tmp_path / "credentials.json"
        # Create manager with custom config to use temp storage
        config = CredentialsConfig(storage_paths=[str(storage_path)])
        manager = CredentialsManager(config=config)
        manager._storage = JsonFileStorage(storage_path)

        # Save expired credentials
        expired_creds = ClaudeCredentials(claudeAiOauth=expired_token)
        await manager.save(expired_creds)

        # Mock OAuth client
        refresh_count = 0

        async def mock_refresh(refresh_token):
            nonlocal refresh_count
            refresh_count += 1
            # Simulate refresh delay
            await asyncio.sleep(0.1)
            return valid_token

        mock_oauth_client = MagicMock()
        mock_oauth_client.refresh_token = AsyncMock(side_effect=mock_refresh)
        manager._oauth_client = mock_oauth_client

        # Launch multiple concurrent requests
        tasks = [manager.get_valid_credentials() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should get the same refreshed token
        assert all(r.claude_ai_oauth.access_token == "valid-token" for r in results)
        # But refresh should only be called once
        assert refresh_count == 1

    @pytest.mark.asyncio
    async def test_refresh_lock_prevents_unnecessary_refresh(
        self, tmp_path, valid_token
    ):
        """Test that the lock prevents refresh if another request already refreshed."""
        storage_path = tmp_path / "credentials.json"
        # Create manager with custom config to use temp storage
        config = CredentialsConfig(storage_paths=[str(storage_path)])
        manager = CredentialsManager(config=config)
        manager._storage = JsonFileStorage(storage_path)

        # Save credentials that are about to expire
        expiring_time = datetime.now(UTC) + timedelta(
            seconds=manager.config.refresh_buffer_seconds - 1
        )
        expiring_ms = int(expiring_time.timestamp() * 1000)

        expiring_creds = ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="expiring-token",
                refreshToken="test-refresh-token",
                expiresAt=expiring_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )
        await manager.save(expiring_creds)

        # Track refresh attempts
        refresh_attempts = []

        async def mock_refresh(refresh_token):
            refresh_attempts.append(datetime.now(UTC))
            # First refresh returns a fresh token
            return valid_token

        mock_oauth_client = MagicMock()
        mock_oauth_client.refresh_token = AsyncMock(side_effect=mock_refresh)
        manager._oauth_client = mock_oauth_client

        # First request should trigger refresh
        await manager.get_valid_credentials()

        # Second request should not trigger refresh (token is now valid)
        await manager.get_valid_credentials()

        # Only one refresh should have occurred
        assert len(refresh_attempts) == 1


class TestAtomicFileOperations:
    """Test atomic file write operations."""

    @pytest.mark.asyncio
    async def test_atomic_write_success(self, tmp_path):
        """Test successful atomic file write."""
        storage_path = tmp_path / "credentials.json"
        storage = JsonFileStorage(storage_path)

        # Create valid credentials
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        creds = ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="test-token",
                refreshToken="test-refresh",
                expiresAt=future_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

        # Save credentials
        assert await storage.save(creds)

        # Check that temp file doesn't exist
        temp_path = storage_path.with_suffix(".tmp")
        assert not temp_path.exists()

        # Check that main file exists with correct content
        assert storage_path.exists()
        loaded = await storage.load()
        assert loaded.claude_ai_oauth.access_token == "test-token"

    @pytest.mark.asyncio
    async def test_atomic_write_cleanup_on_failure(self, tmp_path):
        """Test that temp file is cleaned up on write failure."""
        storage_path = tmp_path / "credentials.json"
        storage = JsonFileStorage(storage_path)

        # Create valid credentials
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        creds = ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="test-token",
                refreshToken="test-refresh",
                expiresAt=future_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

        # Mock json.dump to fail
        with patch("json.dump", side_effect=Exception("Write failed")):
            with pytest.raises(Exception):
                await storage.save(creds)

        # Check that temp file was cleaned up
        temp_path = storage_path.with_suffix(".tmp")
        assert not temp_path.exists()

        # Original file should not exist (first save)
        assert not storage_path.exists()


class TestOAuthTokenValidation:
    """Test OAuth token refresh validation."""

    @pytest.fixture
    def oauth_client(self):
        """Create OAuth client for testing."""
        return OAuthClient()

    @pytest.mark.asyncio
    async def test_refresh_token_validates_access_token(self, oauth_client):
        """Test that refresh validates access_token presence."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            # Missing access_token
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        oauth_client._http_client = mock_http_client

        with pytest.raises(OAuthTokenRefreshError, match="missing access_token"):
            await oauth_client.refresh_token("old-refresh-token")

    @pytest.mark.asyncio
    async def test_refresh_token_handles_missing_expires_in(self, oauth_client):
        """Test that refresh handles missing expires_in with default."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            # Missing expires_in
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        oauth_client._http_client = mock_http_client

        with patch("ccproxy.services.credentials.oauth_client.logger") as mock_logger:
            result = await oauth_client.refresh_token("old-refresh-token")

            # Should use default 1 hour expiration
            assert result.access_token == "new-access-token"
            # Check that warning was logged
            mock_logger.warning.assert_called_once_with(
                "No expires_in in refresh response, using 1 hour default"
            )

            # Token should expire in approximately 1 hour
            time_until_expiry = result.expires_at_datetime - datetime.now(UTC)
            assert 3500 < time_until_expiry.total_seconds() < 3700  # Allow some margin

    @pytest.mark.asyncio
    async def test_refresh_token_preserves_old_refresh_token(self, oauth_client):
        """Test that refresh preserves old refresh token if new one not provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            # No refresh_token in response
            "expires_in": 3600,
        }

        mock_http_client = AsyncMock()
        mock_http_client.post = AsyncMock(return_value=mock_response)
        oauth_client._http_client = mock_http_client

        result = await oauth_client.refresh_token("old-refresh-token")

        # Should preserve the old refresh token
        assert result.access_token == "new-access-token"
        assert result.refresh_token == "old-refresh-token"
