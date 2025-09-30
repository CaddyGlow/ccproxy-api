"""Hook implementation that monitors provider responses for credential failures."""

from __future__ import annotations

from collections.abc import Iterable

from ccproxy.core.plugins.hooks import Hook
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent

from .manager import CredentialBalancerTokenManager


class CredentialBalancerHook(Hook):
    """Hook that routes HTTP lifecycle events to the balancer managers."""

    name = "credential_balancer"
    events = [HookEvent.HTTP_RESPONSE, HookEvent.HTTP_ERROR]
    priority = 550

    def __init__(self, managers: Iterable[CredentialBalancerTokenManager]):
        self._managers: list[CredentialBalancerTokenManager] = list(managers)

    def add_manager(self, manager: CredentialBalancerTokenManager) -> None:
        if manager not in self._managers:
            self._managers.append(manager)

    def remove_manager(self, manager: CredentialBalancerTokenManager) -> None:
        if manager in self._managers:
            self._managers.remove(manager)

    async def __call__(self, context: HookContext) -> None:
        if not self._managers:
            return

        request_id = context.data.get("request_id")
        is_provider = bool(
            context.data.get("is_provider_response")
            or context.data.get("is_provider_request")
        )
        if not request_id or not is_provider:
            return

        status_code = context.data.get("status_code")
        for manager in list(self._managers):
            handled = await manager.handle_response_event(request_id, status_code)
            if handled:
                break


__all__ = ["CredentialBalancerHook"]
