"""HTTP client configuration settings."""

from pydantic import BaseModel, Field


class HTTPSettings(BaseModel):
    """HTTP client configuration settings.

    Controls how the core HTTP client handles compression and other HTTP-level settings.
    """

    compression_enabled: bool = Field(
        default=True,
        description="Enable compression for provider requests (Accept-Encoding header)",
    )

    accept_encoding: str = Field(
        default="gzip, deflate",
        description="Accept-Encoding header value when compression is enabled",
    )

    # Future HTTP settings can be added here:
    # - Connection pooling parameters
    # - Retry policies
    # - Custom headers
    # - Proxy settings
