"""MiniMax's extensible native Messages request envelope."""

from typing import Any, Literal

from pydantic import Field

from ccproxy.llms.formatters import LlmBaseModel


class MiniMaxMessage(LlmBaseModel):
    """Preserve native thinking, tool, image and video blocks without conversion."""

    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class MiniMaxMessageRequest(LlmBaseModel):
    """Validate the envelope; MiniMax validates protocol-specific parameters.

    The adapter forwards the original body, including output_config and complete
    assistant history. The shared schema excludes native thinking/video blocks.
    """

    model: str
    messages: list[MiniMaxMessage]
    max_tokens: int = Field(gt=0)
    stream: bool = False
