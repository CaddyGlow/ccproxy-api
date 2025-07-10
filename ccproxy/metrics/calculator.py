"""
Cost calculation logic for the metrics domain.

This module provides cost calculation functionality for different Claude models
and API usage patterns.
"""

from decimal import Decimal
from typing import Dict, Optional, Tuple

from .models import CostMetric


class CostCalculator:
    """
    Calculates costs for Claude API usage based on token counts and model pricing.
    """
    
    # Claude pricing per 1M tokens (as of 2024)
    # These should be updated based on current Anthropic pricing
    CLAUDE_PRICING: Dict[str, Dict[str, Decimal]] = {
        "claude-3-5-sonnet-20241022": {
            "input": Decimal("3.00"),      # $3.00 per 1M input tokens
            "output": Decimal("15.00"),    # $15.00 per 1M output tokens
            "cache_read": Decimal("0.30"),  # $0.30 per 1M cache read tokens
            "cache_write": Decimal("3.75"), # $3.75 per 1M cache write tokens
        },
        "claude-3-5-haiku-20241022": {
            "input": Decimal("0.25"),      # $0.25 per 1M input tokens
            "output": Decimal("1.25"),     # $1.25 per 1M output tokens
            "cache_read": Decimal("0.03"),  # $0.03 per 1M cache read tokens
            "cache_write": Decimal("0.30"), # $0.30 per 1M cache write tokens
        },
        "claude-3-opus-20240229": {
            "input": Decimal("15.00"),     # $15.00 per 1M input tokens
            "output": Decimal("75.00"),    # $75.00 per 1M output tokens
            "cache_read": Decimal("1.50"),  # $1.50 per 1M cache read tokens
            "cache_write": Decimal("18.75"), # $18.75 per 1M cache write tokens
        },
        "claude-3-sonnet-20240229": {
            "input": Decimal("3.00"),      # $3.00 per 1M input tokens
            "output": Decimal("15.00"),    # $15.00 per 1M output tokens
            "cache_read": Decimal("0.30"),  # $0.30 per 1M cache read tokens
            "cache_write": Decimal("3.75"), # $3.75 per 1M cache write tokens
        },
        "claude-3-haiku-20240307": {
            "input": Decimal("0.25"),      # $0.25 per 1M input tokens
            "output": Decimal("1.25"),     # $1.25 per 1M output tokens
            "cache_read": Decimal("0.03"),  # $0.03 per 1M cache read tokens
            "cache_write": Decimal("0.30"), # $0.30 per 1M cache write tokens
        },
    }
    
    # Default pricing for unknown models (based on Sonnet)
    DEFAULT_PRICING: Dict[str, Decimal] = {
        "input": Decimal("3.00"),
        "output": Decimal("15.00"),
        "cache_read": Decimal("0.30"),
        "cache_write": Decimal("3.75"),
    }
    
    def __init__(self):
        """Initialize the cost calculator."""
        self._custom_pricing: Dict[str, Dict[str, Decimal]] = {}
    
    def add_custom_pricing(self, model: str, pricing: Dict[str, float]) -> None:
        """
        Add custom pricing for a specific model.
        
        Args:
            model: The model name
            pricing: Dictionary with 'input', 'output', 'cache_read', 'cache_write' rates
        """
        self._custom_pricing[model] = {
            key: Decimal(str(value)) for key, value in pricing.items()
        }
    
    def get_model_pricing(self, model: str) -> Dict[str, Decimal]:
        """
        Get pricing information for a specific model.
        
        Args:
            model: The model name
            
        Returns:
            Dictionary with pricing rates per 1M tokens
        """
        # Check custom pricing first
        if model in self._custom_pricing:
            return self._custom_pricing[model]
        
        # Check built-in pricing
        if model in self.CLAUDE_PRICING:
            return self.CLAUDE_PRICING[model]
        
        # Use default pricing
        return self.DEFAULT_PRICING
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        currency: str = "USD"
    ) -> CostMetric:
        """
        Calculate the cost for a given token usage.
        
        Args:
            model: The model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            currency: Currency code (currently only USD supported)
            
        Returns:
            CostMetric with detailed cost breakdown
        """
        pricing = self.get_model_pricing(model)
        
        # Calculate costs (pricing is per 1M tokens)
        input_cost = float(pricing["input"] * Decimal(input_tokens) / Decimal("1000000"))
        output_cost = float(pricing["output"] * Decimal(output_tokens) / Decimal("1000000"))
        cache_read_cost = float(pricing["cache_read"] * Decimal(cache_read_tokens) / Decimal("1000000"))
        cache_write_cost = float(pricing["cache_write"] * Decimal(cache_write_tokens) / Decimal("1000000"))
        
        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost
        
        return CostMetric(
            model=model,
            currency=currency,
            input_cost=input_cost,
            output_cost=output_cost,
            cache_read_cost=cache_read_cost,
            cache_write_cost=cache_write_cost,
            total_cost=total_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    
    def calculate_batch_cost(
        self,
        model: str,
        token_counts: list[Tuple[int, int, int, int]],
        currency: str = "USD"
    ) -> Tuple[float, list[CostMetric]]:
        """
        Calculate costs for multiple requests.
        
        Args:
            model: The model name
            token_counts: List of (input, output, cache_read, cache_write) tuples
            currency: Currency code
            
        Returns:
            Tuple of (total_cost, list_of_cost_metrics)
        """
        cost_metrics = []
        total_cost = 0.0
        
        for input_tokens, output_tokens, cache_read_tokens, cache_write_tokens in token_counts:
            cost_metric = self.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                currency=currency
            )
            cost_metrics.append(cost_metric)
            total_cost += cost_metric.total_cost
        
        return total_cost, cost_metrics
    
    def estimate_cost(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        cache_hit_rate: float = 0.0,
        currency: str = "USD"
    ) -> float:
        """
        Estimate cost for a request before it's made.
        
        Args:
            model: The model name
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            cache_hit_rate: Expected cache hit rate (0.0-1.0)
            currency: Currency code
            
        Returns:
            Estimated cost in the specified currency
        """
        pricing = self.get_model_pricing(model)
        
        # Calculate base costs
        input_cost = float(pricing["input"] * Decimal(estimated_input_tokens) / Decimal("1000000"))
        output_cost = float(pricing["output"] * Decimal(estimated_output_tokens) / Decimal("1000000"))
        
        # Apply cache hit rate to input cost
        cache_read_cost = float(
            pricing["cache_read"] * Decimal(int(estimated_input_tokens * cache_hit_rate)) / Decimal("1000000")
        )
        regular_input_cost = float(
            pricing["input"] * Decimal(int(estimated_input_tokens * (1 - cache_hit_rate))) / Decimal("1000000")
        )
        
        return regular_input_cost + output_cost + cache_read_cost
    
    def get_cost_per_token(self, model: str, token_type: str) -> float:
        """
        Get the cost per token for a specific model and token type.
        
        Args:
            model: The model name
            token_type: Type of token ('input', 'output', 'cache_read', 'cache_write')
            
        Returns:
            Cost per token (not per 1M tokens)
        """
        pricing = self.get_model_pricing(model)
        
        if token_type not in pricing:
            raise ValueError(f"Unknown token type: {token_type}")
        
        return float(pricing[token_type] / Decimal("1000000"))
    
    def compare_model_costs(
        self,
        models: list[str],
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0
    ) -> Dict[str, float]:
        """
        Compare costs across multiple models for the same token usage.
        
        Args:
            models: List of model names to compare
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            
        Returns:
            Dictionary mapping model names to total costs
        """
        costs = {}
        
        for model in models:
            cost_metric = self.calculate_cost(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens
            )
            costs[model] = cost_metric.total_cost
        
        return costs
    
    def get_supported_models(self) -> list[str]:
        """
        Get a list of all supported models.
        
        Returns:
            List of model names with pricing information
        """
        return list(self.CLAUDE_PRICING.keys()) + list(self._custom_pricing.keys())
    
    def validate_token_counts(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0
    ) -> bool:
        """
        Validate that token counts are reasonable.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_read_tokens: Number of cache read tokens
            cache_write_tokens: Number of cache write tokens
            
        Returns:
            True if token counts are valid
        """
        # Check for negative values
        if any(count < 0 for count in [input_tokens, output_tokens, cache_read_tokens, cache_write_tokens]):
            return False
        
        # Check for unreasonably large values (>1M tokens)
        if any(count > 1_000_000 for count in [input_tokens, output_tokens, cache_read_tokens, cache_write_tokens]):
            return False
        
        # Cache write tokens should not exceed input tokens
        if cache_write_tokens > input_tokens:
            return False
        
        # Cache read tokens should not exceed input tokens
        if cache_read_tokens > input_tokens:
            return False
        
        return True