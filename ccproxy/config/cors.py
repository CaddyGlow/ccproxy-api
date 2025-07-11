"""CORS configuration settings."""

from pydantic import BaseModel, Field, field_validator


class CORSSettings(BaseModel):
    """CORS-specific configuration settings."""

    origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )

    credentials: bool = Field(
        default=True,
        description="CORS allow credentials",
    )

    methods: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed methods",
    )

    headers: list[str] = Field(
        default_factory=lambda: ["*"],
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
