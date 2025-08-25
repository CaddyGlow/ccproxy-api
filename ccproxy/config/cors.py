"""CORS configuration settings."""

from pydantic import BaseModel, Field, field_validator


class CORSSettings(BaseModel):
    """CORS-specific configuration settings."""

    origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ],
        description="CORS allowed origins (avoid using '*' for security)",
    )

    credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )

    methods: list[str] = Field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="CORS allowed methods",
    )

    headers: list[str] = Field(
        default_factory=lambda: [
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        description="CORS allowed headers",
    )

    origin_regex: str | None = Field(
        default=None,
        description="CORS origin regex pattern",
    )

    expose_headers: list[str] = Field(
        default_factory=list,
        description="CORS exposed headers",
    )

    max_age: int = Field(
        default=600,
        description="CORS preflight max age in seconds",
        ge=0,
    )

    @field_validator("origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("methods", mode="before")
    @classmethod
    def validate_cors_methods(cls, v: str | list[str]) -> list[str]:
        """Parse CORS methods from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [method.strip().upper() for method in v.split(",") if method.strip()]
        return [method.upper() for method in v]

    @field_validator("headers", mode="before")
    @classmethod
    def validate_cors_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    @field_validator("expose_headers", mode="before")
    @classmethod
    def validate_cors_expose_headers(cls, v: str | list[str]) -> list[str]:
        """Parse CORS expose headers from string or list."""
        if isinstance(v, str):
            # Split comma-separated string
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    def is_origin_allowed(self, origin: str | None) -> bool:
        """Check if an origin is allowed by the CORS policy.

        Args:
            origin: The origin to check (from request Origin header)

        Returns:
            bool: True if origin is allowed, False otherwise
        """
        if not origin:
            return False

        # Check against explicit origins list
        if origin in self.origins:
            return True

        # Check if wildcard is explicitly configured
        if "*" in self.origins:
            return True

        # Check against regex pattern if configured
        if self.origin_regex:
            import re

            try:
                return bool(re.match(self.origin_regex, origin))
            except re.error:
                return False

        return False

    def get_allowed_origin(self, request_origin: str | None) -> str | None:
        """Get the appropriate CORS origin value for response headers.

        Args:
            request_origin: The origin from the request

        Returns:
            str | None: The origin to set in Access-Control-Allow-Origin header,
                       or None if origin is not allowed
        """
        if not request_origin:
            return None

        if self.is_origin_allowed(request_origin):
            # Return specific origin instead of wildcard for security
            # Only return "*" if explicitly configured and credentials are False
            if "*" in self.origins and not self.credentials:
                return "*"
            else:
                return request_origin

        return None
