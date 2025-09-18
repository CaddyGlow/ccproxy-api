"""Shared base model for all LLM API models."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class LlmBaseModel(BaseModel):
    """Base model for all LLM API models with proper JSON serialization.

    Excludes None values and empty collections to match API conventions.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields
    )

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Override to exclude empty collections as well as None values."""
        # First get the data with None values excluded
        data = super().model_dump(exclude_none=True, **kwargs)

        # Filter out empty collections (lists, dicts, sets)
        filtered_data = {}
        for key, value in data.items():
            if isinstance(value, list | dict | set) and len(value) == 0:
                # Skip empty collections
                continue
            filtered_data[key] = value

        return filtered_data
