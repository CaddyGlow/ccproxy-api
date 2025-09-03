"""Server configuration settings."""

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    """Server-specific configuration settings."""

    host: str = Field(
        default="127.0.0.1",
        description="Server host address",
    )

    port: int = Field(
        default=8000,
        description="Server port number",
        ge=1,
        le=65535,
    )

    workers: int = Field(
        default=1,
        description="Number of worker processes",
        ge=1,
        le=32,
    )

    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )

    use_terminal_permission_handler: bool = Field(
        default=False,
        description="Enable terminal UI for permission prompts. Set to False to use external handler via SSE (not implemented)",
    )

    bypass_mode: bool = Field(
        default=False,
        description="Enable bypass mode for testing (uses mock responses instead of real API calls)",
    )
