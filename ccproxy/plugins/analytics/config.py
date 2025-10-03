from pydantic import BaseModel, Field


class AnalyticsPluginConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable analytics routes")
