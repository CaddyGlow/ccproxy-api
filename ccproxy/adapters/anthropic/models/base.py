"""Common base model for all Anthropic models."""

from pydantic import BaseModel, ConfigDict


class AnthropicBaseModel(BaseModel):
    """Base model for all Anthropic-specific models.

    Sets common configuration including extra="allow" to handle
    additional fields that may be present in Anthropic API responses.
    """

    model_config = ConfigDict(extra="allow")
