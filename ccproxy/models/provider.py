"""Provider configuration models."""

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a provider plugin."""

    name: str = Field(..., description="Provider name")
    base_url: str = Field(..., description="Base URL for the provider API")
    supports_streaming: bool = Field(
        default=False, description="Whether the provider supports streaming"
    )
    requires_auth: bool = Field(
        default=True, description="Whether the provider requires authentication"
    )
    auth_type: str | None = Field(
        default=None, description="Authentication type (bearer, api_key, etc.)"
    )
    models: list[str] = Field(
        default_factory=list, description="List of supported models"
    )
