"""Common base model for all OpenAI models."""

from pydantic import BaseModel, ConfigDict


class OpenAIBaseModel(BaseModel):
    """Base model for all OpenAI-specific models.

    Sets common configuration including extra="allow" to handle
    additional fields that may be present in OpenAI API responses.
    """

    model_config = ConfigDict(extra="allow")
