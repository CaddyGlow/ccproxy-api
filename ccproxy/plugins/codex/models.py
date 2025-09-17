"""Codex plugin local CLI health models and detection models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ccproxy.llms.models import anthropic as anthropic_models


class CodexCliStatus(str, Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    BINARY_FOUND_BUT_ERRORS = "binary_found_but_errors"
    TIMEOUT = "timeout"
    ERROR = "error"


class CodexCliInfo(BaseModel):
    status: CodexCliStatus
    version: str | None = None
    binary_path: str | None = None
    version_output: str | None = None
    error: str | None = None
    return_code: str | None = None


class CodexHeaders(BaseModel):
    """Pydantic model for Codex CLI headers extraction with field aliases."""

    session_id: str = Field(
        alias="session_id",
        description="Codex session identifier",
        default="",
    )
    originator: str = Field(
        description="Codex originator identifier",
        default="codex_cli_rs",
    )
    openai_beta: str = Field(
        alias="openai-beta",
        description="OpenAI beta features",
        default="responses=experimental",
    )
    version: str = Field(
        description="Codex CLI version",
        default="0.21.0",
    )
    chatgpt_account_id: str = Field(
        alias="chatgpt-account-id",
        description="ChatGPT account identifier",
        default="",
    )

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def to_headers_dict(self) -> dict[str, str]:
        """Convert to headers dictionary for HTTP forwarding with proper case."""
        headers = {}

        # Map field names to proper HTTP header names
        header_mapping = {
            "session_id": "session_id",
            "originator": "originator",
            "openai_beta": "openai-beta",
            "version": "version",
            "chatgpt_account_id": "chatgpt-account-id",
        }

        for field_name, header_name in header_mapping.items():
            value = getattr(self, field_name, None)
            if value is not None and value != "":
                headers[header_name] = value

        return headers


class CodexInstructionsData(BaseModel):
    """Extracted Codex instructions information."""

    instructions_field: Annotated[
        str,
        Field(
            description="Complete instructions field as detected from Codex CLI, preserving exact text content"
        ),
    ]

    model_config = ConfigDict(extra="forbid")


class CodexCacheData(BaseModel):
    """Cached Codex CLI detection data with version tracking."""

    codex_version: Annotated[str, Field(description="Codex CLI version")]
    headers: Annotated[
        dict[str, str],
        Field(description="Captured headers (lowercase keys) in insertion order"),
    ]
    body_json: Annotated[
        dict[str, Any] | None,
        Field(description="Captured request body as JSON if parseable", default=None),
    ] = None
    method: Annotated[
        str | None, Field(description="Captured HTTP method", default=None)
    ] = None
    url: Annotated[str | None, Field(description="Captured full URL", default=None)] = (
        None
    )
    path: Annotated[
        str | None, Field(description="Captured request path", default=None)
    ] = None
    query_params: Annotated[
        dict[str, str] | None,
        Field(description="Captured query parameters", default=None),
    ] = None
    cached_at: Annotated[
        datetime,
        Field(
            description="Cache timestamp",
            default_factory=lambda: datetime.now(UTC),
        ),
    ] = None  # type: ignore # Pydantic handles this via default_factory

    model_config = ConfigDict(extra="forbid")


class CodexMessage(BaseModel):
    """Message format for Codex requests."""

    role: Annotated[Literal["user", "assistant"], Field(description="Message role")]
    content: Annotated[str, Field(description="Message content")]


class CodexRequest(BaseModel):
    """OpenAI Codex completion request model."""

    model: Annotated[str, Field(description="Model name (e.g., gpt-5)")] = "gpt-5"
    instructions: Annotated[
        str | None, Field(description="System instructions for the model")
    ] = None
    messages: Annotated[list[CodexMessage], Field(description="Conversation messages")]
    stream: Annotated[bool, Field(description="Whether to stream the response")] = True

    model_config = ConfigDict(
        extra="allow"
    )  # Allow additional fields for compatibility


class CodexResponse(BaseModel):
    """OpenAI Codex completion response model."""

    id: Annotated[str, Field(description="Response ID")]
    model: Annotated[str, Field(description="Model used for completion")]
    content: Annotated[str, Field(description="Generated content")]
    finish_reason: Annotated[
        str | None, Field(description="Reason the response finished")
    ] = None
    usage: Annotated[
        anthropic_models.Usage | None, Field(description="Token usage information")
    ] = None

    model_config = ConfigDict(
        extra="allow"
    )  # Allow additional fields for compatibility


class CodexAuthData(TypedDict, total=False):
    """Authentication data for Codex/OpenAI provider.

    Attributes:
        access_token: Bearer token for OpenAI API authentication
        chatgpt_account_id: Account ID for ChatGPT session-based requests
    """

    access_token: str | None
    chatgpt_account_id: str | None
