"""Codex plugin local CLI health models and detection models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts


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

    model_config = ConfigDict(extra="allow")


class CodexCacheData(BaseModel):
    """Cached Codex CLI detection data with version tracking."""

    codex_version: Annotated[str, Field(description="Codex CLI version")]
    headers: Annotated[
        DetectedHeaders,
        Field(
            description="Captured headers (lowercase keys) in insertion order",
            default_factory=DetectedHeaders,
        ),
    ]
    prompts: Annotated[
        DetectedPrompts,
        Field(description="Captured prompt metadata", default_factory=DetectedPrompts),
    ]
    body_json: Annotated[
        dict[str, Any] | None,
        Field(
            description="Legacy captured request body (deprecated)",
            default=None,
            exclude=True,
        ),
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
    cached_at: datetime = Field(
        description="Cache timestamp",
        default_factory=lambda: datetime.now(UTC),
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_format(cls, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            return values

        if "prompts" not in values:
            legacy_body = values.get("body_json")
            if legacy_body is not None:
                values["prompts"] = DetectedPrompts.from_body(legacy_body)
                cls._log_legacy_usage("body_json")

        return values

    @field_validator("headers", mode="before")
    @classmethod
    def _validate_headers(cls, value: Any) -> DetectedHeaders:
        if isinstance(value, DetectedHeaders):
            return value
        if isinstance(value, dict):
            return DetectedHeaders(value)
        if value is None:
            cls._log_legacy_usage("missing_headers")
            return DetectedHeaders()
        raise TypeError("headers must be a mapping of strings")

    @field_validator("prompts", mode="before")
    @classmethod
    def _validate_prompts(cls, value: Any) -> DetectedPrompts:
        if isinstance(value, DetectedPrompts):
            return value
        if isinstance(value, dict):
            return DetectedPrompts.from_body(value)
        if value is None:
            return DetectedPrompts()
        raise TypeError("prompts must be derived from a mapping")

    @field_serializer("headers")
    def _serialize_headers(self, headers: DetectedHeaders) -> dict[str, str]:
        return headers.as_dict()

    @field_serializer("prompts")
    def _serialize_prompts(self, prompts: DetectedPrompts) -> dict[str, Any]:
        raw = prompts.raw or {}
        if not isinstance(raw, dict):
            raw = {}
        if prompts.instructions and "instructions" not in raw:
            raw = dict(raw)
            raw["instructions"] = prompts.instructions
        if prompts.system is not None and "system" not in raw:
            raw = dict(raw)
            raw["system"] = prompts.system
        return raw

    @staticmethod
    def _log_legacy_usage(reason: str) -> None:
        try:
            from ccproxy.core.logging import get_plugin_logger

            logger = get_plugin_logger()
            logger.debug(
                "legacy_detection_cache_format",
                plugin="codex",
                reason=reason,
            )
        except Exception:  # pragma: no cover - logging best-effort only
            pass


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
