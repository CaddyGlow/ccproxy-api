"""Claude API plugin local CLI health models and detection models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Annotated, Any, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from ccproxy.models.detection import DetectedHeaders, DetectedPrompts


class ClaudeCliStatus(str, Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    BINARY_FOUND_BUT_ERRORS = "binary_found_but_errors"
    TIMEOUT = "timeout"
    ERROR = "error"


class ClaudeCliInfo(BaseModel):
    status: ClaudeCliStatus
    version: str | None = None
    binary_path: str | None = None
    version_output: str | None = None
    error: str | None = None
    return_code: str | None = None


class ClaudeAgentHeaders(BaseModel):
    """Pydantic model for Claude CLI headers extraction with field aliases."""

    anthropic_beta: str = Field(
        alias="anthropic-beta",
        description="Anthropic beta features",
        default="claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
    )
    anthropic_version: str = Field(
        alias="anthropic-version",
        description="Anthropic API version",
        default="2023-06-01",
    )
    anthropic_dangerous_direct_browser_access: str = Field(
        alias="anthropic-dangerous-direct-browser-access",
        description="Browser access flag",
        default="true",
    )
    x_app: str = Field(
        alias="x-app", description="Application identifier", default="cli"
    )
    user_agent: str = Field(
        alias="user-agent",
        description="User agent string",
        default="claude-cli/1.0.60 (external, cli)",
    )
    x_stainless_lang: str = Field(
        alias="x-stainless-lang", description="SDK language", default="js"
    )
    x_stainless_retry_count: str = Field(
        alias="x-stainless-retry-count", description="Retry count", default="0"
    )
    x_stainless_timeout: str = Field(
        alias="x-stainless-timeout", description="Request timeout", default="60"
    )
    x_stainless_package_version: str = Field(
        alias="x-stainless-package-version",
        description="Package version",
        default="0.55.1",
    )
    x_stainless_os: str = Field(
        alias="x-stainless-os", description="Operating system", default="Linux"
    )
    x_stainless_arch: str = Field(
        alias="x-stainless-arch", description="Architecture", default="x64"
    )
    x_stainless_runtime: str = Field(
        alias="x-stainless-runtime", description="Runtime", default="node"
    )
    x_stainless_runtime_version: str = Field(
        alias="x-stainless-runtime-version",
        description="Runtime version",
        default="v24.3.0",
    )

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    def to_headers_dict(self) -> dict[str, str]:
        """Convert to headers dictionary for HTTP forwarding with proper case."""
        headers = {}

        # Map field names to proper HTTP header names
        header_mapping = {
            "anthropic_beta": "anthropic-beta",
            "anthropic_version": "anthropic-version",
            "anthropic_dangerous_direct_browser_access": "anthropic-dangerous-direct-browser-access",
            "x_app": "x-app",
            "user_agent": "User-Agent",
            "x_stainless_lang": "X-Stainless-Lang",
            "x_stainless_retry_count": "X-Stainless-Retry-Count",
            "x_stainless_timeout": "X-Stainless-Timeout",
            "x_stainless_package_version": "X-Stainless-Package-Version",
            "x_stainless_os": "X-Stainless-OS",
            "x_stainless_arch": "X-Stainless-Arch",
            "x_stainless_runtime": "X-Stainless-Runtime",
            "x_stainless_runtime_version": "X-Stainless-Runtime-Version",
        }

        for field_name, header_name in header_mapping.items():
            value = getattr(self, field_name, None)
            if value is not None:
                headers[header_name] = value

        return headers


class SystemPromptData(BaseModel):
    """Extracted system prompt information."""

    system_field: Annotated[
        str | list[dict[str, Any]],
        Field(
            description="Complete system field as detected from Claude CLI, preserving exact structure including type, text, and cache_control"
        ),
    ]

    model_config = ConfigDict(extra="forbid")


class ClaudeCacheData(BaseModel):
    """Cached Claude CLI detection data with version tracking."""

    claude_version: Annotated[str, Field(description="Claude CLI version")]
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
                plugin="claude_api",
                reason=reason,
            )
        except Exception:  # pragma: no cover - logging best-effort only
            pass


class ClaudeAPIAuthData(TypedDict, total=False):
    """Authentication data for Claude API provider.

    Attributes:
        access_token: Bearer token for Anthropic Claude API authentication
    """

    access_token: str | None
