"""OAuth-specific models for authentication."""

from datetime import datetime

from pydantic import BaseModel, Field


class OAuthState(BaseModel):
    """OAuth state information for pending flows."""

    code_verifier: str = Field(..., description="PKCE code verifier")
    custom_paths: list[str] | None = Field(None, description="Custom credential paths")
    completed: bool = Field(default=False, description="Whether the flow is completed")
    success: bool = Field(default=False, description="Whether the flow was successful")
    error: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request parameters."""

    code: str | None = Field(None, description="Authorization code")
    state: str | None = Field(None, description="State parameter")
    error: str | None = Field(None, description="OAuth error")
    error_description: str | None = Field(None, description="OAuth error description")


class OAuthTokenRequest(BaseModel):
    """OAuth token exchange request."""

    grant_type: str = Field(default="authorization_code")
    code: str = Field(..., description="Authorization code")
    redirect_uri: str = Field(..., description="Redirect URI")
    client_id: str = Field(..., description="Client ID")
    code_verifier: str = Field(..., description="PKCE code verifier")


class OAuthTokenResponse(BaseModel):
    """OAuth token exchange response."""

    access_token: str = Field(..., description="Access token")
    refresh_token: str | None = Field(None, description="Refresh token")
    expires_in: int | None = Field(None, description="Token expiration in seconds")
    scope: str | None = Field(None, description="Granted scopes")
    subscription_type: str | None = Field(None, description="Subscription type")
    token_type: str = Field(default="Bearer", description="Token type")
