"""Tests for Claude credentials models."""

from datetime import UTC, datetime, timedelta

from claude_code_proxy.services.credentials import (
    AccountInfo,
    ClaudeCredentials,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)


class TestOAuthToken:
    """Test OAuth token model."""

    def test_oauth_token_parsing(self):
        """Test parsing OAuth token from JSON data."""
        token_data = {
            "accessToken": "test-access-token",
            "refreshToken": "test-refresh-token",
            "expiresAt": 1751896667201,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }

        token = OAuthToken.model_validate(token_data)

        assert token.access_token == "test-access-token"
        assert token.refresh_token == "test-refresh-token"
        assert token.expires_at == 1751896667201
        assert token.scopes == ["user:inference", "user:profile"]
        assert token.subscription_type == "max"

    def test_token_expiry_check(self):
        """Test token expiry checking."""
        # Create a token that expires in 1 hour
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": future_ms,
            }
        )

        assert not token.is_expired

        # Create an expired token
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        expired_token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": past_ms,
            }
        )

        assert expired_token.is_expired

    def test_expires_at_datetime_conversion(self):
        """Test conversion of expires_at to datetime."""
        # Use a known timestamp
        timestamp_ms = 1751896667201
        expected_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": timestamp_ms,
            }
        )

        assert token.expires_at_datetime == expected_dt


class TestClaudeCredentials:
    """Test Claude credentials model."""

    def test_credentials_parsing(self):
        """Test parsing full credentials from JSON."""
        creds_data = {
            "claudeAiOauth": {
                "accessToken": "test-access-token",
                "refreshToken": "test-refresh-token",
                "expiresAt": 1751896667201,
                "scopes": ["user:inference"],
                "subscriptionType": "pro",
            }
        }

        credentials = ClaudeCredentials.model_validate(creds_data)

        assert credentials.claude_ai_oauth.access_token == "test-access-token"
        assert credentials.claude_ai_oauth.refresh_token == "test-refresh-token"
        assert credentials.claude_ai_oauth.expires_at == 1751896667201
        assert credentials.claude_ai_oauth.scopes == ["user:inference"]
        assert credentials.claude_ai_oauth.subscription_type == "pro"


class TestUserProfile:
    """Test user profile models."""

    def test_organization_info_creation(self):
        """Test OrganizationInfo model creation."""
        org = OrganizationInfo(uuid="org-123", name="Test Organization")
        assert org.uuid == "org-123"
        assert org.name == "Test Organization"

    def test_account_info_creation(self):
        """Test AccountInfo model creation."""
        account = AccountInfo(uuid="user-456", email_address="test@example.com")
        assert account.uuid == "user-456"
        assert account.email_address == "test@example.com"

    def test_user_profile_creation(self):
        """Test UserProfile model creation with all fields."""
        org = OrganizationInfo(uuid="org-123", name="Test Organization")
        account = AccountInfo(uuid="user-456", email_address="test@example.com")
        profile = UserProfile(organization=org, account=account)

        assert profile.organization == org
        assert profile.account == account

    def test_user_profile_optional_fields(self):
        """Test UserProfile with optional fields."""
        profile = UserProfile()
        assert profile.organization is None
        assert profile.account is None
