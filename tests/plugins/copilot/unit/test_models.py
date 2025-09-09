"""Unit tests for Copilot plugin models."""

from datetime import datetime

from ccproxy.plugins.copilot.models import (
    CopilotCacheData,
    CopilotCliInfo,
    CopilotEmbeddingRequest,
    CopilotHealthResponse,
    CopilotQuotaSnapshot,
    CopilotTokenStatus,
    CopilotUserInternalResponse,
)


class TestCopilotEmbeddingRequest:
    """Test cases for CopilotEmbeddingRequest."""

    def test_basic_initialization(self) -> None:
        """Test basic embedding request initialization."""
        request = CopilotEmbeddingRequest(
            input="Hello, world!",
        )

        assert request.input == "Hello, world!"
        assert request.model == "text-embedding-ada-002"
        assert request.user is None

    def test_with_custom_model(self) -> None:
        """Test initialization with custom model."""
        request = CopilotEmbeddingRequest(
            input="Test text",
            model="custom-embedding-model",
            user="test-user",
        )

        assert request.input == "Test text"
        assert request.model == "custom-embedding-model"
        assert request.user == "test-user"

    def test_list_input(self) -> None:
        """Test with list of strings as input."""
        texts = ["First text", "Second text", "Third text"]
        request = CopilotEmbeddingRequest(input=texts)

        assert request.input == texts
        assert request.model == "text-embedding-ada-002"


class TestCopilotHealthResponse:
    """Test cases for CopilotHealthResponse."""

    def test_basic_initialization(self) -> None:
        """Test basic health response initialization."""
        response = CopilotHealthResponse(status="healthy")

        assert response.status == "healthy"
        assert response.provider == "copilot"
        assert isinstance(response.timestamp, datetime)

    def test_unhealthy_status(self) -> None:
        """Test unhealthy status response."""
        details = {"error": "Connection failed"}
        response = CopilotHealthResponse(
            status="unhealthy",
            details=details,
        )

        assert response.status == "unhealthy"
        assert response.details == details


class TestCopilotTokenStatus:
    """Test cases for CopilotTokenStatus."""

    def test_valid_token(self) -> None:
        """Test valid token status."""
        expires_at = datetime.now()
        status = CopilotTokenStatus(
            valid=True,
            expires_at=expires_at,
            account_type="pro",
            copilot_access=True,
            username="testuser",
        )

        assert status.valid is True
        assert status.expires_at == expires_at
        assert status.account_type == "pro"
        assert status.copilot_access is True
        assert status.username == "testuser"

    def test_invalid_token(self) -> None:
        """Test invalid token status."""
        status = CopilotTokenStatus(
            valid=False,
            account_type="free",
            copilot_access=False,
        )

        assert status.valid is False
        assert status.expires_at is None
        assert status.account_type == "free"
        assert status.copilot_access is False
        assert status.username is None


class TestCopilotQuotaSnapshot:
    """Test cases for CopilotQuotaSnapshot."""

    def test_basic_initialization(self) -> None:
        """Test basic quota snapshot initialization."""
        snapshot = CopilotQuotaSnapshot(
            entitlement=1000,
            overage_count=0,
            overage_permitted=True,
            percent_remaining=75.5,
            quota_id="chat-quota",
            quota_remaining=755.0,
            remaining=755,
            unlimited=False,
            timestamp_utc="2024-01-01T00:00:00Z",
        )

        assert snapshot.entitlement == 1000
        assert snapshot.overage_count == 0
        assert snapshot.overage_permitted is True
        assert snapshot.percent_remaining == 75.5
        assert snapshot.quota_id == "chat-quota"
        assert snapshot.quota_remaining == 755.0
        assert snapshot.remaining == 755
        assert snapshot.unlimited is False
        assert snapshot.timestamp_utc == "2024-01-01T00:00:00Z"


class TestCopilotUserInternalResponse:
    """Test cases for CopilotUserInternalResponse."""

    def test_basic_initialization(self) -> None:
        """Test basic user internal response initialization."""
        quota_snapshots = {
            "chat": CopilotQuotaSnapshot(
                entitlement=1000,
                overage_count=0,
                overage_permitted=True,
                percent_remaining=80.0,
                quota_id="chat",
                quota_remaining=800.0,
                remaining=800,
                unlimited=False,
                timestamp_utc="2024-01-01T00:00:00Z",
            )
        }

        response = CopilotUserInternalResponse(
            access_type_sku="copilot_pro",
            analytics_tracking_id="track-123",
            can_signup_for_limited=True,
            chat_enabled=True,
            copilot_plan="pro",
            quota_reset_date="2024-01-31",
            quota_snapshots=quota_snapshots,
            quota_reset_date_utc="2024-01-31T23:59:59Z",
        )

        assert response.access_type_sku == "copilot_pro"
        assert response.analytics_tracking_id == "track-123"
        assert response.can_signup_for_limited is True
        assert response.chat_enabled is True
        assert response.copilot_plan == "pro"
        assert response.quota_reset_date == "2024-01-31"
        assert "chat" in response.quota_snapshots
        assert response.quota_reset_date_utc == "2024-01-31T23:59:59Z"


class TestCopilotCacheData:
    """Test cases for CopilotCacheData."""

    def test_basic_initialization(self) -> None:
        """Test basic cache data initialization."""
        cache_data = CopilotCacheData(
            cli_available=True,
            cli_version="2.40.1",
            auth_status="authenticated",
            username="testuser",
        )

        assert cache_data.cli_available is True
        assert cache_data.cli_version == "2.40.1"
        assert cache_data.auth_status == "authenticated"
        assert cache_data.username == "testuser"
        assert isinstance(cache_data.last_check, datetime)

    def test_cli_unavailable(self) -> None:
        """Test cache data with CLI unavailable."""
        cache_data = CopilotCacheData(cli_available=False)

        assert cache_data.cli_available is False
        assert cache_data.cli_version is None
        assert cache_data.auth_status is None
        assert cache_data.username is None


class TestCopilotCliInfo:
    """Test cases for CopilotCliInfo."""

    def test_available_and_authenticated(self) -> None:
        """Test CLI info for available and authenticated CLI."""
        cli_info = CopilotCliInfo(
            available=True,
            version="2.40.1",
            authenticated=True,
            username="testuser",
        )

        assert cli_info.available is True
        assert cli_info.version == "2.40.1"
        assert cli_info.authenticated is True
        assert cli_info.username == "testuser"
        assert cli_info.error is None

    def test_unavailable_with_error(self) -> None:
        """Test CLI info for unavailable CLI with error."""
        cli_info = CopilotCliInfo(
            available=False,
            error="GitHub CLI not found in PATH",
        )

        assert cli_info.available is False
        assert cli_info.version is None
        assert cli_info.authenticated is False
        assert cli_info.username is None
        assert cli_info.error == "GitHub CLI not found in PATH"

    def test_available_but_not_authenticated(self) -> None:
        """Test CLI info for available but not authenticated CLI."""
        cli_info = CopilotCliInfo(
            available=True,
            version="2.39.0",
            authenticated=False,
        )

        assert cli_info.available is True
        assert cli_info.version == "2.39.0"
        assert cli_info.authenticated is False
        assert cli_info.username is None
        assert cli_info.error is None
