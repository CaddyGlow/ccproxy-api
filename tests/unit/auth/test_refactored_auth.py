"""Tests for refactored authentication components."""

from datetime import UTC, datetime, timedelta

from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo


class TestBaseModels:
    """Test base authentication models."""

    def test_base_token_info_is_expired(self):
        """Test that is_expired computed field works correctly."""

        class TestToken(BaseTokenInfo):
            test_expires_at: datetime

            @property
            def access_token_value(self) -> str:
                return "test_token"

            @property
            def expires_at_datetime(self) -> datetime:
                return self.test_expires_at

        # Test expired token
        expired_token = TestToken(
            test_expires_at=datetime.now(UTC) - timedelta(hours=1)
        )
        assert expired_token.is_expired is True

        # Test valid token
        valid_token = TestToken(test_expires_at=datetime.now(UTC) + timedelta(hours=1))
        assert valid_token.is_expired is False

    def test_base_profile_info(self):
        """Test BaseProfileInfo model."""
        profile = BaseProfileInfo(
            account_id="test_id",
            provider_type="test_provider",
            email="test@example.com",
            display_name="Test User",
            extras={"custom": "data"},
        )

        assert profile.account_id == "test_id"
        assert profile.provider_type == "test_provider"
        assert profile.email == "test@example.com"
        assert profile.display_name == "Test User"
        assert profile.extras == {"custom": "data"}
