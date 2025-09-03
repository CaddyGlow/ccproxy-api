"""Pricing service exceptions."""


class PricingError(Exception):
    """Base exception for pricing-related errors."""

    pass


class PricingDataNotLoadedError(PricingError):
    """Raised when pricing data has not been loaded yet."""

    def __init__(
        self,
        message: str = "Pricing data not loaded yet - cost calculation unavailable",
    ):
        self.message = message
        super().__init__(self.message)


class ModelPricingNotFoundError(PricingError):
    """Raised when pricing for a specific model is not found."""

    def __init__(self, model: str, message: str | None = None):
        self.model = model
        self.message = message or f"No pricing data available for model '{model}'"
        super().__init__(self.message)


class PricingServiceDisabledError(PricingError):
    """Raised when pricing service is disabled."""

    def __init__(self, message: str = "Pricing service is disabled"):
        self.message = message
        super().__init__(self.message)
