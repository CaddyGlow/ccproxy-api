"""Plugin entry point for the credential balancer."""

from __future__ import annotations

from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import (
    PluginContext,
    PluginManifest,
    SystemPluginFactory,
    SystemPluginRuntime,
)
from ccproxy.services.auth_registry import AuthManagerRegistry

from .config import CredentialBalancerSettings
from .hook import CredentialBalancerHook
from .manager import CredentialBalancerTokenManager


logger = get_plugin_logger()


class CredentialBalancerRuntime(SystemPluginRuntime):
    """Runtime responsible for registering auth managers and hooks."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self._registrations: list[tuple[str, CredentialBalancerTokenManager]] = []
        self._hook: CredentialBalancerHook | None = None
        self._registry: AuthManagerRegistry | None = None

    async def _on_initialize(self) -> None:
        await super()._on_initialize()
        if not self.context:
            raise RuntimeError("Context not set")

        config = self.context.get("config")
        if not isinstance(config, CredentialBalancerSettings):
            logger.debug("credential_balancer_using_default_config")
            config = CredentialBalancerSettings()

        if not config.enabled:
            logger.info("credential_balancer_disabled")
            return

        if not config.providers:
            logger.warning("credential_balancer_no_providers_configured")
            return

        service_container = self.context.get("service_container")
        if not service_container:
            raise RuntimeError("Service container unavailable for credential balancer")

        registry = service_container.get_auth_manager_registry()
        self._registry = registry

        base_logger = self.context.get("logger") or get_plugin_logger(__name__)
        managers: list[CredentialBalancerTokenManager] = []

        for pool in config.providers:
            manager_name = pool.manager_name
            if manager_name is None:
                raise ValueError(
                    f"Credential balancer pool '{pool.provider}' missing manager name"
                )
            manager_logger = base_logger.bind(pool=manager_name)
            # Use async factory to create manager with composed AuthManagers
            manager = await CredentialBalancerTokenManager.create(
                pool, logger=manager_logger
            )
            registry.register_instance(manager_name, manager)
            managers.append(manager)
            self._registrations.append((manager_name, manager))
            logger.info(
                "credential_balancer_manager_registered",
                manager=manager_name,
                provider=pool.provider,
                strategy=pool.strategy.value,
                credentials=len(pool.credentials),
            )

        if managers:
            hook_registry = self.context.get("hook_registry")
            if not hook_registry:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry

            if hook_registry:
                hook = CredentialBalancerHook(managers)
                hook_registry.register(hook)
                self._hook = hook
                logger.debug("credential_balancer_hook_registered")
            else:
                logger.warning("credential_balancer_hook_registry_missing")

    async def _on_shutdown(self) -> None:
        await super()._on_shutdown()
        if self.context and self._hook:
            hook_registry = self.context.get("hook_registry")
            if not hook_registry:
                app = self.context.get("app")
                if app and hasattr(app.state, "hook_registry"):
                    hook_registry = app.state.hook_registry
            if hook_registry:
                hook_registry.unregister(self._hook)
                logger.debug("credential_balancer_hook_unregistered")
        self._hook = None

        if self._registry:
            for name, _ in self._registrations:
                try:
                    self._registry.unregister(name)
                except Exception:
                    logger.debug(
                        "credential_balancer_registry_unregistration_failed",
                        manager=name,
                    )
        self._registrations.clear()


class CredentialBalancerFactory(SystemPluginFactory):
    """Factory for the credential balancer plugin."""

    def __init__(self) -> None:
        manifest = PluginManifest(
            name="credential_balancer",
            version="0.1.0",
            description="Rotate across multiple credential files for upstream providers",
            is_provider=False,
            config_class=CredentialBalancerSettings,
        )
        super().__init__(manifest)

    def create_runtime(self) -> CredentialBalancerRuntime:
        return CredentialBalancerRuntime(self.manifest)

    def create_context(self, core_services: Any) -> PluginContext:
        context = super().create_context(core_services)
        return context


factory = CredentialBalancerFactory()

__all__ = ["CredentialBalancerFactory", "CredentialBalancerRuntime", "factory"]
