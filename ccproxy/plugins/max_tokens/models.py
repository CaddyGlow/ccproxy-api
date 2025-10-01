"""Pydantic models for max_tokens plugin."""

from pydantic import BaseModel, Field


class ModelTokenLimits(BaseModel):
    """Token limits for a specific model."""

    max_output_tokens: int = Field(
        ..., ge=1, description="Maximum output tokens for the model"
    )
    max_input_tokens: int | None = Field(
        default=None, ge=1, description="Maximum input tokens for the model"
    )


class TokenLimitsData(BaseModel):
    """Complete token limits data for all models."""

    models: dict[str, ModelTokenLimits] = Field(
        default_factory=dict, description="Model name to token limits mapping"
    )

    def get_max_output_tokens(self, model_name: str) -> int | None:
        """Get maximum output tokens for a model."""
        model_limits = self.models.get(model_name)
        return model_limits.max_output_tokens if model_limits else None

    def get_max_input_tokens(self, model_name: str) -> int | None:
        """Get maximum input tokens for a model."""
        model_limits = self.models.get(model_name)
        return model_limits.max_input_tokens if model_limits else None

    def has_model(self, model_name: str) -> bool:
        """Check if model limits exist for the given model."""
        return model_name in self.models


class MaxTokensModification(BaseModel):
    """Information about max_tokens modification made by the plugin."""

    original_max_tokens: int | None = Field(
        description="Original max_tokens value from request"
    )
    new_max_tokens: int | None = Field(
        description="New max_tokens value after modification"
    )
    model: str = Field(description="Model name")
    reason: str = Field(description="Reason for modification")

    def was_modified(self) -> bool:
        """Check if max_tokens was modified."""
        return self.original_max_tokens != self.new_max_tokens
