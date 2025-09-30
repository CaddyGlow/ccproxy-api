"""Factory for creating AuthManager instances from credential sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager
from ccproxy.core.logging import TraceBoundLogger, get_plugin_logger

from .config import CredentialFile, CredentialManager, CredentialSource
from .file_adapter import FileBackedAuthManager


if TYPE_CHECKING:
    from ccproxy.services.auth_registry import AuthManagerRegistry


logger = get_plugin_logger(__name__)


class AuthManagerFactory:
    """Creates AuthManager instances from credential source configurations."""

    def __init__(
        self,
        auth_registry: AuthManagerRegistry | None = None,
        *,
        logger: TraceBoundLogger | None = None,
    ) -> None:
        """Initialize auth manager factory.

        Args:
            auth_registry: Auth manager registry for resolving manager keys
            logger: Optional logger for this factory
        """
        self._auth_registry = auth_registry
        self._logger = logger or get_plugin_logger(__name__)

    async def create_from_source(
        self,
        source: CredentialSource,
        provider: str,
    ) -> AuthManager:
        """Create AuthManager instance from credential source configuration.

        Args:
            source: Credential source configuration (file or manager)
            provider: Provider name for this credential

        Returns:
            AuthManager instance

        Raises:
            AuthenticationError: If manager creation fails
        """
        if isinstance(source, CredentialFile):
            return await self._create_file_backed_manager(source, provider)
        elif isinstance(source, CredentialManager):
            return await self._create_provider_manager(source)
        else:
            raise AuthenticationError(
                f"Unsupported credential source type: {type(source).__name__}"
            )

    async def _create_file_backed_manager(
        self,
        source: CredentialFile,
        provider: str,
    ) -> FileBackedAuthManager:
        """Create file-backed auth manager for legacy token snapshots.

        Args:
            source: File credential configuration
            provider: Provider name

        Returns:
            FileBackedAuthManager instance
        """
        self._logger.debug(
            "creating_file_backed_manager",
            path=str(source.path),
            provider=provider,
            label=source.resolved_label,
        )
        return FileBackedAuthManager(
            path=source.path,
            provider=provider,
            logger=self._logger.bind(
                credential_type="file", label=source.resolved_label
            ),
        )

    async def _create_provider_manager(
        self,
        source: CredentialManager,
    ) -> AuthManager:
        """Create provider-specific auth manager from registry.

        Args:
            source: Manager credential configuration

        Returns:
            AuthManager instance from registry

        Raises:
            AuthenticationError: If manager key not found in registry
        """
        if self._auth_registry is None:
            raise AuthenticationError(
                f"Auth registry not available for manager key: {source.manager_key}"
            )

        self._logger.debug(
            "creating_provider_manager",
            manager_key=source.manager_key,
            label=source.resolved_label,
        )

        manager = await self._auth_registry.get(source.manager_key)
        if manager is None:
            raise AuthenticationError(
                f"Auth manager not found in registry: {source.manager_key}"
            )

        self._logger.info(
            "provider_manager_created",
            manager_key=source.manager_key,
            label=source.resolved_label,
            manager_type=type(manager).__name__,
        )
        return manager  # type: ignore[no-any-return]


__all__ = ["AuthManagerFactory"]
