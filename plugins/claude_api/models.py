"""Claude API plugin local CLI health models."""

from enum import Enum

from pydantic import BaseModel


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
