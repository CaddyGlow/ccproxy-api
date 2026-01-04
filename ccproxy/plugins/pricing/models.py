"""Pydantic models for pricing data validation and type safety."""

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_serializer,
    field_validator,
)


class ModelPricing(BaseModel):
    """Pricing information for a single Claude model.

    All costs are in USD per 1 million tokens.
    """

    input: Decimal = Field(..., ge=0, description="Input token cost per 1M tokens")
    output: Decimal = Field(..., ge=0, description="Output token cost per 1M tokens")
    cache_read: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cache read cost per 1M tokens"
    )
    cache_write: Decimal = Field(
        default=Decimal("0"), ge=0, description="Cache write cost per 1M tokens"
    )

    @field_validator("*", mode="before")
    @classmethod
    def convert_to_decimal(cls, v: Any) -> Decimal:
        """Convert numeric values to Decimal for precision."""
        if isinstance(v, int | float | str):
            return Decimal(str(v))
        if isinstance(v, Decimal):
            return v
        raise TypeError(f"Cannot convert {type(v)} to Decimal")

    @field_serializer("input", "output", "cache_read", "cache_write")
    def serialize_decimal(self, value: Decimal) -> float:
        """Serialize Decimal fields as float for JSON compatibility."""
        return float(value)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class PricingData(RootModel[dict[str, ModelPricing]]):
    """Complete pricing data for all Claude models.

    This is a wrapper around a dictionary of model name to ModelPricing
    that provides dict-like access while maintaining type safety.
    """

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Iterate over model names."""
        return iter(self.root)

    def __getitem__(self, model_name: str) -> ModelPricing:
        """Get pricing for a specific model."""
        return self.root[model_name]

    def __contains__(self, model_name: str) -> bool:
        """Check if model exists in pricing data."""
        return model_name in self.root

    def __len__(self) -> int:
        """Get number of models in pricing data."""
        return len(self.root)

    def items(self) -> Iterator[tuple[str, ModelPricing]]:
        """Get model name and pricing pairs."""
        return iter(self.root.items())

    def keys(self) -> Iterator[str]:
        """Get model names."""
        return iter(self.root.keys())

    def values(self) -> Iterator[ModelPricing]:
        """Get pricing objects."""
        return iter(self.root.values())

    def get(
        self, model_name: str, default: ModelPricing | None = None
    ) -> ModelPricing | None:
        """Get pricing for a model with optional default."""
        return self.root.get(model_name, default)

    def model_names(self) -> list[str]:
        """Get list of all model names."""
        return list(self.root.keys())
