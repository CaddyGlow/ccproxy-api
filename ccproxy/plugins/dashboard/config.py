from pydantic import BaseModel, Field


class DashboardPluginConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable dashboard routes")
    mount_static: bool = Field(
        default=True, description="Mount /dashboard/assets static files if present"
    )
