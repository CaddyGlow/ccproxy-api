from pydantic import BaseModel, Field


class AnalyticsPluginConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable analytics routes")
    route_prefix: str = Field(default="/logs", description="Route prefix for logs API")
