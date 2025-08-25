"""Codex provider plugin v2 implementation."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.plugins import (
    PluginContext,
    PluginManifest,
    ProviderPluginFactory,
    ProviderPluginRuntime,
    RouteSpec,
)
from plugins.codex.adapter import CodexAdapter
from plugins.codex.config import CodexSettings
from plugins.codex.detection_service import CodexDetectionService
from plugins.codex.routes import router as codex_router


logger = get_plugin_logger()


class CodexRuntime(ProviderPluginRuntime):
    """Runtime for Codex provider plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)
        self.config: CodexSettings | None = None
        self.auth_manager: Any | None = None

    async def _on_initialize(self) -> None:
        """Initialize the Codex provider plugin."""
        if not self.context:
            raise RuntimeError("Context not set")

        # Get configuration
        config = self.context.get("config")
        if not isinstance(config, CodexSettings):
            logger.warning("plugin_no_config")
            # Use default config if none provided
            config = CodexSettings()
            logger.info("plugin_using_default_config")
        self.config = config

        # Get auth manager from context
        self.auth_manager = self.context.get("credentials_manager")

        # Call parent to initialize adapter and detection service
        await super()._on_initialize()

        # Check CLI status
        if self.detection_service:
            version = self.detection_service.get_version()
            cli_path = self.detection_service.get_cli_path()

            if not cli_path:
                logger.warning(
                    "cli_detection_completed",
                    cli_available=False,
                    version=None,
                    cli_path=None,
                    source="unknown",
                )

        # Get CLI info for consolidated logging (only for successful detection)
        cli_info = {}
        if self.detection_service and self.detection_service.get_cli_path():
            cli_info.update(
                {
                    "cli_available": True,
                    "cli_version": self.detection_service.get_version(),
                    "cli_path": self.detection_service.get_cli_path(),
                    "cli_source": "package_manager",
                }
            )

        logger.debug(
            "plugin_initialized",
            status="initialized",
            has_credentials=self.auth_manager is not None,
            has_adapter=self.adapter is not None,
            has_detection=self.detection_service is not None,
            category="plugin",
            **cli_info,
        )

    async def get_profile_info(self) -> dict[str, Any] | None:
        """Get Codex-specific profile information from stored credentials."""
        try:
            import base64
            import json

            # Get access token from stored credentials
            if not self.auth_manager:
                return None

            access_token = await self.auth_manager.get_access_token()
            if not access_token:
                return None

            # For OpenAI/Codex, extract info from JWT token
            parts = access_token.split(".")
            if len(parts) != 3:
                return None

            # Decode JWT payload
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))

            profile_info = {}

            # Extract OpenAI-specific information
            openai_auth = payload.get("https://api.openai.com/auth", {})
            if openai_auth:
                if "email" in payload:
                    profile_info["email"] = payload["email"]
                    profile_info["email_verified"] = payload.get(
                        "email_verified", False
                    )

                if openai_auth.get("chatgpt_plan_type"):
                    profile_info["plan_type"] = openai_auth["chatgpt_plan_type"].upper()

                if openai_auth.get("chatgpt_user_id"):
                    profile_info["user_id"] = openai_auth["chatgpt_user_id"]

                # Subscription info
                if openai_auth.get("chatgpt_subscription_active_start"):
                    profile_info["subscription_start"] = openai_auth[
                        "chatgpt_subscription_active_start"
                    ]
                if openai_auth.get("chatgpt_subscription_active_until"):
                    profile_info["subscription_until"] = openai_auth[
                        "chatgpt_subscription_active_until"
                    ]

                # Organizations
                orgs = openai_auth.get("organizations", [])
                if orgs:
                    for org in orgs:
                        if org.get("is_default"):
                            profile_info["organization"] = org.get("title", "Unknown")
                            profile_info["organization_role"] = org.get(
                                "role", "member"
                            )
                            profile_info["organization_id"] = org.get("id", "Unknown")
                            break

            return profile_info if profile_info else None

        except Exception as e:
            logger.debug(f"Failed to get Codex profile info: {e}")
            return None

    async def get_auth_summary(self) -> dict[str, Any]:
        """Get detailed authentication status."""
        if not self.auth_manager:
            return {"auth": "not_configured"}

        try:
            auth_status = await self.auth_manager.get_auth_status()
            summary = {"auth": "not_configured"}

            if auth_status.get("auth_configured"):
                if auth_status.get("token_available"):
                    summary["auth"] = "authenticated"
                    if "time_remaining" in auth_status:
                        summary["auth_expires"] = auth_status["time_remaining"]
                    if "token_expired" in auth_status:
                        summary["auth_expired"] = auth_status["token_expired"]
                else:
                    summary["auth"] = "no_token"
            else:
                summary["auth"] = "not_configured"

            return summary
        except Exception as e:
            logger.warning(
                "codex_auth_status_error", error=str(e), exc_info=e, category="auth"
            )
            return {"auth": "status_error"}

    async def _get_health_details(self) -> dict[str, Any]:
        """Get health check details."""
        details = await super()._get_health_details()

        # Add Codex-specific details
        if self.config:
            details.update(
                {
                    "base_url": self.config.base_url,
                    "supports_streaming": self.config.supports_streaming,
                    "models": self.config.models,
                }
            )

        # Add authentication status
        if self.auth_manager:
            try:
                auth_status = await self.auth_manager.get_auth_status()
                details["auth_configured"] = auth_status.get("auth_configured", False)
                details["token_available"] = auth_status.get("token_available", False)
            except Exception as e:
                details["auth_error"] = str(e)

        return details


class CodexFactory(ProviderPluginFactory):
    """Factory for Codex provider plugin."""

    def __init__(self) -> None:
        """Initialize factory with manifest."""
        # Create manifest with static declarations
        manifest = PluginManifest(
            name="codex",
            version="1.0.0",
            description="OpenAI Codex provider plugin with OAuth authentication and format conversion",
            is_provider=True,
            config_class=CodexSettings,
            dependencies=[
                "oauth_codex"
            ],  # Depends on OAuth Codex plugin for authentication
            routes=[
                RouteSpec(
                    router=codex_router,
                    prefix="/api/codex",
                    tags=["provider", "codex"],
                ),
            ],
            # OAuth functionality now provided by oauth_codex plugin
        )

        # Initialize with manifest
        super().__init__(manifest)

    def create_runtime(self) -> CodexRuntime:
        """Create runtime instance."""
        return CodexRuntime(self.manifest)

    def create_adapter(self, context: PluginContext) -> CodexAdapter:
        """Create the Codex adapter."""
        proxy_service = context.get("proxy_service")
        auth_manager = context.get("credentials_manager")
        detection_service = context.get("detection_service")
        http_client = context.get("http_client")
        logger_instance = context.get("logger")

        if not proxy_service:
            raise RuntimeError("ProxyService is required for Codex adapter")
        if not auth_manager:
            raise RuntimeError("Auth manager is required for Codex adapter")
        if not detection_service:
            raise RuntimeError("Detection service is required for Codex adapter")

        return CodexAdapter(
            proxy_service=proxy_service,
            auth_manager=auth_manager,
            detection_service=detection_service,
            http_client=http_client,
            logger=logger_instance,
        )

    def create_detection_service(self, context: PluginContext) -> CodexDetectionService:
        """Create the Codex detection service."""
        settings = context.get("settings")
        if not settings:
            raise RuntimeError("Settings are required for Codex detection service")

        cli_service = context.get("cli_detection_service")
        return CodexDetectionService(settings, cli_service)

    def create_credentials_manager(self, context: PluginContext) -> Any:
        """Create the Codex credentials manager.

        Note: OAuth functionality is now provided by the oauth_codex plugin.
        The token manager will look up OAuth providers from the registry as needed.
        """
        from plugins.codex.auth.manager import CodexTokenManager

        return CodexTokenManager()


# Export the factory instance
factory = CodexFactory()
