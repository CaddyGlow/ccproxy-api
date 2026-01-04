"""Generate scaffold files for new CCProxy plugins."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from textwrap import dedent


class PluginTemplateType(str, Enum):
    """Supported plugin template variants."""

    SYSTEM = "system"
    PROVIDER = "provider"
    AUTH = "auth"


@dataclass(frozen=True)
class TemplateContext:
    """Precomputed naming helpers for template rendering."""

    plugin_name: str
    pascal_name: str
    description: str
    version: str

    @property
    def config_class(self) -> str:
        return f"{self.pascal_name}Config"

    @property
    def runtime_class(self) -> str:
        return f"{self.pascal_name}Runtime"

    @property
    def factory_class(self) -> str:
        return f"{self.pascal_name}Factory"

    @property
    def adapter_class(self) -> str:
        return f"{self.pascal_name}Adapter"

    @property
    def detection_class(self) -> str:
        return f"{self.pascal_name}DetectionService"

    @property
    def oauth_provider_class(self) -> str:
        return f"{self.pascal_name}OAuthProvider"


def build_plugin_scaffold(
    plugin_name: str,
    description: str,
    version: str,
    template_type: PluginTemplateType,
    *,
    include_tests: bool = False,
) -> dict[str, str]:
    """Return a mapping of relative file paths to scaffold contents."""

    pascal_name = _to_pascal_case(plugin_name)
    ctx = TemplateContext(
        plugin_name=plugin_name,
        pascal_name=pascal_name,
        description=description,
        version=version,
    )

    builders: dict[
        PluginTemplateType, Callable[[TemplateContext, bool], dict[str, str]]
    ] = {
        PluginTemplateType.SYSTEM: _build_system_template,
        PluginTemplateType.PROVIDER: _build_provider_template,
        PluginTemplateType.AUTH: _build_auth_template,
    }

    builder = builders.get(template_type)
    if builder is None:
        raise ValueError(f"Unsupported template type: {template_type}")

    return builder(ctx, include_tests)


def _build_system_template(ctx: TemplateContext, include_tests: bool) -> dict[str, str]:
    """Build files for a system plugin scaffold."""

    files: dict[str, str] = {
        "__init__.py": dedent(
            f'''
            """Runtime package for {ctx.pascal_name} plugin."""

            from .plugin import factory

            __all__ = ["factory"]
            '''
        ).strip()
        + "\n",
        "config.py": dedent(
            f'''
            """Configuration for the {ctx.pascal_name} plugin."""

            from pydantic import BaseModel, Field


            class {ctx.config_class}(BaseModel):
                """Runtime configuration toggles for {ctx.plugin_name}."""

                enabled: bool = Field(
                    default=True,
                    description="Enable the plugin once configured.",
                )
            '''
        ).strip()
        + "\n",
        "plugin.py": dedent(
            f'''
            """System plugin runtime for {ctx.plugin_name}."""

            from __future__ import annotations

            from ccproxy.core.logging import get_plugin_logger
            from ccproxy.core.plugins import (
                PluginManifest,
                SystemPluginFactory,
                SystemPluginRuntime,
            )

            from .config import {ctx.config_class}


            logger = get_plugin_logger()


            class {ctx.runtime_class}(SystemPluginRuntime):
                """Runtime implementation for {ctx.plugin_name}."""

                def __init__(self, manifest: PluginManifest) -> None:
                    super().__init__(manifest)
                    self.config: {ctx.config_class} | None = None

                async def _on_initialize(self) -> None:
                    await super()._on_initialize()
                    if not self.context:
                        raise RuntimeError("Plugin context is not available")

                    try:
                        self.config = self.context.get({ctx.config_class})
                    except ValueError:
                        self.config = {ctx.config_class}()
                        logger.debug(
                            "plugin_using_default_config",
                            plugin=self.name,
                        )

                    if not self.config.enabled:
                        logger.info("plugin_disabled", plugin=self.name)
                        return

                    logger.info("plugin_initialized", plugin=self.name)


            class {ctx.factory_class}(SystemPluginFactory):
                """Factory for the {ctx.plugin_name} system plugin."""

                def __init__(self) -> None:
                    manifest = PluginManifest(
                        name="{ctx.plugin_name}",
                        version="{ctx.version}",
                        description="{ctx.description}",
                        is_provider=False,
                        config_class={ctx.config_class},
                    )
                    super().__init__(manifest)

                def create_runtime(self) -> {ctx.runtime_class}:
                    return {ctx.runtime_class}(self.manifest)


            factory = {ctx.factory_class}()
            '''
        ).strip()
        + "\n",
        "README.md": dedent(
            f"""
            # {ctx.pascal_name} Plugin

            Generated with `ccproxy plugins scaffold {ctx.plugin_name}`.

            ## Next steps

            - Update `config.py` with any required settings.
            - Implement runtime logic in `plugin.py`.
            - Add tests under `tests/` to cover the behaviour you introduce.
            """
        ).strip()
        + "\n",
    }

    if include_tests:
        files.update(
            {
                "tests/__init__.py": "\n",
                f"tests/test_{ctx.plugin_name}.py": dedent(
                    f'''
                    """Smoke tests for the {ctx.plugin_name} plugin scaffold."""

                    from ccproxy.plugins.{ctx.plugin_name}.plugin import factory


                    def test_manifest_defaults() -> None:
                        manifest = factory.get_manifest()
                        assert manifest.name == "{ctx.plugin_name}"
                        assert manifest.version == "{ctx.version}"
                    '''
                ).strip()
                + "\n",
            }
        )

    return files


def _build_provider_template(
    ctx: TemplateContext, include_tests: bool
) -> dict[str, str]:
    """Build files for a provider plugin scaffold."""

    files: dict[str, str] = {
        "__init__.py": dedent(
            f'''
            """Provider plugin package for {ctx.pascal_name}."""

            from .plugin import factory

            __all__ = ["factory"]
            '''
        ).strip()
        + "\n",
        "config.py": dedent(
            f'''
            """Configuration for the {ctx.pascal_name} provider."""

            from pydantic import BaseModel, Field, HttpUrl


            class {ctx.config_class}(BaseModel):
                """Runtime configuration options for {ctx.plugin_name}."""

                enabled: bool = Field(
                    default=False,
                    description="Enable once adapter logic has been implemented.",
                )
                base_url: HttpUrl = Field(
                    default="https://api.example.com",
                    description="Upstream API base URL.",
                )
                supports_streaming: bool = Field(
                    default=False,
                    description="Set to true when streaming is implemented.",
                )
                requires_auth: bool = Field(
                    default=True,
                    description="Whether requests need authentication headers.",
                )
            '''
        ).strip()
        + "\n",
        "adapter.py": dedent(
            f'''
            """HTTP adapter stub for {ctx.plugin_name}."""

            from __future__ import annotations

            from typing import Any

            from fastapi import HTTPException, Request
            from starlette.responses import JSONResponse

            from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
            from ccproxy.streaming import DeferredStreaming


            class {ctx.adapter_class}(BaseHTTPAdapter):
                """Minimal adapter that surfaces implementation TODOs."""

                async def handle_request(
                    self, request: Request
                ) -> JSONResponse | DeferredStreaming:
                    raise HTTPException(
                        status_code=503,
                        detail="{ctx.plugin_name} adapter is not implemented yet.",
                    )

                async def handle_streaming(
                    self,
                    request: Request,
                    endpoint: str,
                    **_: Any,
                ) -> DeferredStreaming:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "Streaming for {ctx.plugin_name} is not implemented yet."
                        ),
                    )

                async def cleanup(self) -> None:
                    """Adapter cleanup hook."""

                    return None
            '''
        ).strip()
        + "\n",
        "plugin.py": dedent(
            f'''
            """Provider plugin runtime for {ctx.plugin_name}."""

            from __future__ import annotations

            from ccproxy.core.logging import get_plugin_logger
            from ccproxy.core.plugins import (
                BaseProviderPluginFactory,
                PluginContext,
                PluginManifest,
                ProviderPluginRuntime,
            )
            from ccproxy.models.provider import ProviderConfig

            from .adapter import {ctx.adapter_class}
            from .config import {ctx.config_class}


            logger = get_plugin_logger()


            class {ctx.runtime_class}(ProviderPluginRuntime):
                """Runtime implementation for {ctx.plugin_name}."""

                def __init__(self, manifest: PluginManifest) -> None:
                    super().__init__(manifest)
                    self.config: {ctx.config_class} | None = None

                async def _on_initialize(self) -> None:
                    if not self.context:
                        raise RuntimeError("Plugin context is not available")

                    try:
                        self.config = self.context.get({ctx.config_class})
                    except ValueError:
                        self.config = {ctx.config_class}()

                    if not self.config.enabled:
                        logger.info("plugin_disabled", plugin=self.name)
                        return

                    await super()._on_initialize()
                    logger.info("plugin_initialized", plugin=self.name)

                async def _on_validate(self) -> bool:
                    config = self.config or {ctx.config_class}()
                    if not config.enabled:
                        return True
                    return await super()._on_validate()


            class {ctx.factory_class}(BaseProviderPluginFactory):
                """Factory for the {ctx.plugin_name} provider plugin."""

                plugin_name = "{ctx.plugin_name}"
                plugin_description = "{ctx.description}"
                plugin_version = "{ctx.version}"
                runtime_class = {ctx.runtime_class}
                adapter_class = {ctx.adapter_class}
                config_class = {ctx.config_class}

                async def create_adapter(
                    self, context: PluginContext
                ) -> {ctx.adapter_class}:
                    config = context.get({ctx.config_class})
                    provider_config = ProviderConfig(
                        name=self.plugin_name,
                        base_url=str(config.base_url),
                        supports_streaming=config.supports_streaming,
                        requires_auth=config.requires_auth,
                    )

                    return {ctx.adapter_class}(
                        config=provider_config,
                        auth_manager=None,
                        http_pool_manager=context.http_pool_manager,
                        streaming_handler=context.streaming_handler,
                        format_registry=context.format_registry,
                    )

                def create_detection_service(
                    self, context: PluginContext
                ) -> None:
                    config = context.get({ctx.config_class})
                    if not config.enabled:
                        return None
                    return None

                async def create_credentials_manager(
                    self, context: PluginContext
                ) -> None:
                    config = context.get({ctx.config_class})
                    if not config.enabled:
                        return None
                    return None


            factory = {ctx.factory_class}()
            '''
        ).strip()
        + "\n",
        "README.md": dedent(
            f"""
            # {ctx.pascal_name} Provider Plugin

            Generated with `ccproxy plugins scaffold {ctx.plugin_name} --type provider`.

            ## Next steps

            - Fill in `adapter.py` with provider-specific HTTP logic.
            - Wire authentication and detection in `plugin.py`.
            - Provide plugin-specific tests before enabling the plugin.
            """
        ).strip()
        + "\n",
    }

    if include_tests:
        files.update(
            {
                "tests/__init__.py": "\n",
                f"tests/test_{ctx.plugin_name}_manifest.py": dedent(
                    f'''
                    """Manifest smoke tests for the {ctx.plugin_name} provider."""

                    from ccproxy.plugins.{ctx.plugin_name}.plugin import factory


                    def test_manifest_marks_provider() -> None:
                        manifest = factory.get_manifest()
                        assert manifest.is_provider is True
                    '''
                ).strip()
                + "\n",
            }
        )

    return files


def _build_auth_template(ctx: TemplateContext, include_tests: bool) -> dict[str, str]:
    """Build files for an auth provider plugin scaffold."""

    files: dict[str, str] = {
        "__init__.py": dedent(
            f'''
            """Auth provider plugin package for {ctx.pascal_name}."""

            from .plugin import factory

            __all__ = ["factory"]
            '''
        ).strip()
        + "\n",
        "config.py": dedent(
            f'''
            """Configuration for the {ctx.pascal_name} auth provider."""

            from pydantic import BaseModel, Field


            class {ctx.config_class}(BaseModel):
                """Runtime options for {ctx.plugin_name} OAuth."""

                enabled: bool = Field(
                    default=False,
                    description="Enable once OAuth endpoints are implemented.",
                )
                client_id: str = Field(
                    default="your-client-id",
                    description="OAuth client identifier.",
                )
                client_secret: str = Field(
                    default="your-client-secret",
                    description="OAuth client secret.",
                )
                auth_base_url: str = Field(
                    default="https://auth.example.com",
                    description="Base URL for OAuth authorization endpoints.",
                )
                token_url: str = Field(
                    default="https://auth.example.com/token",
                    description="Token exchange endpoint.",
                )
            '''
        ).strip()
        + "\n",
        "provider.py": dedent(
            f'''
            """OAuth provider stub for {ctx.plugin_name}."""

            from __future__ import annotations

            from typing import Any

            from ccproxy.auth.oauth.registry import OAuthProviderInfo, OAuthProviderProtocol


            class {ctx.oauth_provider_class}(OAuthProviderProtocol):
                """Skeleton OAuth provider awaiting implementation."""

                def __init__(self, *, display_name: str) -> None:
                    self._display_name = display_name

                @property
                def provider_name(self) -> str:
                    return "{ctx.plugin_name}"

                @property
                def provider_display_name(self) -> str:
                    return self._display_name

                @property
                def supports_pkce(self) -> bool:
                    return True

                async def get_authorization_url(
                    self,
                    state: str,
                    code_verifier: str | None = None,
                    redirect_uri: str | None = None,
                ) -> str:
                    raise NotImplementedError("Build authorization URL logic")

                async def handle_callback(
                    self,
                    code: str,
                    state: str,
                    code_verifier: str | None = None,
                    redirect_uri: str | None = None,
                ) -> Any:
                    raise NotImplementedError("Exchange auth code for tokens")

                async def refresh_access_token(self, refresh_token: str) -> Any:
                    raise NotImplementedError("Refresh tokens using the provider API")

                async def revoke_token(self, token: str) -> None:
                    raise NotImplementedError("Revoke an issued token if supported")

                async def get_profile(self, access_token: str) -> Any:
                    raise NotImplementedError("Fetch account profile information")

                def describe(self) -> OAuthProviderInfo:
                    return OAuthProviderInfo(
                        name=self.provider_name,
                        display_name=self.provider_display_name,
                        description="{ctx.description}",
                        plugin_name="{ctx.plugin_name}",
                    )
            '''
        ).strip()
        + "\n",
        "plugin.py": dedent(
            f'''
            """Auth provider plugin runtime for {ctx.plugin_name}."""

            from __future__ import annotations

            from ccproxy.core.logging import get_plugin_logger
            from ccproxy.core.plugins import (
                AuthProviderPluginFactory,
                AuthProviderPluginRuntime,
                PluginContext,
                PluginManifest,
            )

            from .config import {ctx.config_class}
            from .provider import {ctx.oauth_provider_class}


            logger = get_plugin_logger()


            class {ctx.runtime_class}(AuthProviderPluginRuntime):
                """Runtime implementation for {ctx.plugin_name} OAuth."""

                def __init__(self, manifest: PluginManifest) -> None:
                    super().__init__(manifest)
                    self.config: {ctx.config_class} | None = None

                async def _on_initialize(self) -> None:
                    if not self.context:
                        raise RuntimeError("Plugin context is not available")

                    try:
                        self.config = self.context.get({ctx.config_class})
                    except ValueError:
                        self.config = {ctx.config_class}()

                    if not self.config.enabled:
                        logger.info("plugin_disabled", plugin=self.name)
                        return

                    await super()._on_initialize()
                    logger.info("plugin_initialized", plugin=self.name)


            class {ctx.factory_class}(AuthProviderPluginFactory):
                """Factory for the {ctx.plugin_name} OAuth plugin."""

                cli_safe = True

                def __init__(self) -> None:
                    manifest = PluginManifest(
                        name="{ctx.plugin_name}",
                        version="{ctx.version}",
                        description="{ctx.description}",
                        is_provider=True,
                        config_class={ctx.config_class},
                    )
                    super().__init__(manifest)

                def create_context(self, service_container: object) -> PluginContext:
                    context = super().create_context(service_container)
                    config = context.get({ctx.config_class})

                    if config.enabled:
                        context["auth_provider"] = self.create_auth_provider(context)

                    return context

                def create_runtime(self) -> {ctx.runtime_class}:
                    return {ctx.runtime_class}(self.manifest)

                def create_auth_provider(
                    self, context: PluginContext | None = None
                ) -> {ctx.oauth_provider_class}:
                    return {ctx.oauth_provider_class}(display_name="{ctx.pascal_name}")


            factory = {ctx.factory_class}()
            '''
        ).strip()
        + "\n",
        "README.md": dedent(
            f"""
            # {ctx.pascal_name} OAuth Plugin

            Generated with `ccproxy plugins scaffold {ctx.plugin_name} --type auth`.

            ## Next steps

            - Implement the OAuth flow in `provider.py`.
            - Wire storage and credential handling inside `plugin.py`.
            - Provide CLI guidance for user sign-in.
            """
        ).strip()
        + "\n",
    }

    if include_tests:
        files.update(
            {
                "tests/__init__.py": "\n",
                f"tests/test_{ctx.plugin_name}_auth.py": dedent(
                    f'''
                    """Manifest smoke tests for the {ctx.plugin_name} auth plugin."""

                    from ccproxy.plugins.{ctx.plugin_name}.plugin import factory


                    def test_manifest_marks_provider() -> None:
                        manifest = factory.get_manifest()
                        assert manifest.is_provider is True
                    '''
                ).strip()
                + "\n",
            }
        )

    return files


def _to_pascal_case(name: str) -> str:
    """Convert snake-case or kebab-case names to PascalCase."""

    parts = re.split(r"[_\-\s]+", name)
    return "".join(part.capitalize() for part in parts if part)
