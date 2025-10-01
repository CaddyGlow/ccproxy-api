"""Factory for creating AuthManager instances from credential sources."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager
from ccproxy.core.logging import TraceBoundLogger, get_plugin_logger

from .config import CredentialManager


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
        source: CredentialManager,
        provider: str,
    ) -> AuthManager:
        """Create AuthManager instance from credential source configuration.

        Args:
            source: Manager credential configuration
            provider: Provider name for this credential (unused, kept for compatibility)

        Returns:
            AuthManager instance

        Raises:
            AuthenticationError: If manager creation fails
        """
        return await self._create_provider_manager(source)

    async def _create_provider_manager(
        self,
        source: CredentialManager,
    ) -> AuthManager:
        """Create provider-specific auth manager.

        Args:
            source: Manager credential configuration

        Returns:
            AuthManager instance

        Raises:
            AuthenticationError: If manager creation fails
        """
        # Check if custom file path is specified (already expanded by validator)
        custom_file = str(source.file.resolve()) if source.file else None

        # Direct class specification approach
        if source.manager_class:
            return await self._create_manager_from_class_name(
                source.manager_class,
                source.storage_class,
                custom_file,
                source.resolved_label,
                source.config,
            )

        # Registry lookup approach
        if source.manager_key:
            return await self._create_manager_from_registry(
                source.manager_key,
                custom_file,
                source.resolved_label,
            )

        raise AuthenticationError(
            "Neither manager_class nor manager_key specified in credential source"
        )

    async def _create_manager_from_registry(
        self,
        manager_key: str,
        custom_file: str | None,
        label: str,
    ) -> AuthManager:
        """Create manager using registry lookup.

        Args:
            manager_key: Registry key
            custom_file: Optional custom file path
            label: Label for logging

        Returns:
            AuthManager instance

        Raises:
            AuthenticationError: If manager not found or creation fails
        """
        if self._auth_registry is None:
            raise AuthenticationError(
                f"Auth registry not available for manager key: {manager_key}"
            )

        if custom_file:
            # Create manager with custom storage
            return await self._create_manager_with_custom_file(
                manager_key,
                custom_file,
                label,
            )

        # Standard registry lookup
        self._logger.debug(
            "creating_provider_manager_from_registry",
            manager_key=manager_key,
            label=label,
        )

        manager = await self._auth_registry.get(manager_key)
        if manager is None:
            raise AuthenticationError(
                f"Auth manager not found in registry: {manager_key}"
            )

        self._logger.info(
            "provider_manager_created_from_registry",
            manager_key=manager_key,
            label=label,
            manager_type=type(manager).__name__,
        )
        return manager  # type: ignore[no-any-return]

    async def _create_manager_from_class_name(
        self,
        manager_class_name: str,
        storage_class_name: str | None,
        custom_file: str | None,
        label: str,
        config: dict[str, Any] | None = None,
    ) -> AuthManager:
        """Create manager by dynamically importing class.

        Args:
            manager_class_name: Fully qualified manager class name
            storage_class_name: Fully qualified storage class name (required if custom_file specified)
            custom_file: Optional custom file path
            label: Label for logging
            config: Additional configuration options for storage and manager

        Returns:
            AuthManager instance

        Raises:
            AuthenticationError: If class cannot be imported or instantiated
        """
        config = config or {}

        self._logger.debug(
            "creating_manager_from_class_name",
            manager_class=manager_class_name,
            storage_class=storage_class_name,
            custom_file=custom_file,
            label=label,
            config_keys=list(config.keys()),
        )

        # Import manager class
        try:
            manager_class = self._import_class(manager_class_name)
        except Exception as exc:
            raise AuthenticationError(
                f"Failed to import manager class '{manager_class_name}': {exc}"
            ) from exc

        # Create storage if custom file specified
        storage = None
        if custom_file:
            if not storage_class_name:
                raise AuthenticationError(
                    "storage_class is required when using custom file with manager_class"
                )

            try:
                storage_class = self._import_class(storage_class_name)
                # custom_file is already expanded and resolved by config validator
                custom_path = Path(custom_file)

                # Extract storage-specific config options
                storage_kwargs: dict[str, Any] = {"storage_path": custom_path}
                if "enable_backups" in config:
                    storage_kwargs["enable_backups"] = bool(config["enable_backups"])

                storage = storage_class(**storage_kwargs)
            except Exception as exc:
                raise AuthenticationError(
                    f"Failed to create storage from '{storage_class_name}': {exc}"
                ) from exc

        # Create manager instance with config options
        try:
            # Check if we have advanced config options that need direct __init__ call
            has_advanced_config = (
                "credentials_ttl" in config or "refresh_grace_seconds" in config
            )

            if has_advanced_config:
                # Use direct __init__ to pass ttl/grace parameters
                # These are supported by BaseTokenManager but not exposed in create() methods
                init_kwargs: dict[str, Any] = {"storage": storage}
                if "credentials_ttl" in config:
                    init_kwargs["credentials_ttl"] = float(config["credentials_ttl"])
                if "refresh_grace_seconds" in config:
                    init_kwargs["refresh_grace_seconds"] = float(
                        config["refresh_grace_seconds"]
                    )

                manager = manager_class(**init_kwargs)
            elif hasattr(manager_class, "create"):
                # Use async create() method for standard instantiation
                manager = (
                    await manager_class.create(storage=storage)
                    if storage
                    else await manager_class.create()
                )
            else:
                raise AuthenticationError(
                    f"Manager class {manager_class.__name__} does not have 'create' method"
                )
        except FileNotFoundError as exc:
            # Clean warning for missing credential files
            file_path = custom_file or "default location"
            self._logger.warning(
                "credential_file_not_found",
                label=label,
                file_path=file_path,
                manager_class=manager_class_name,
            )
            raise AuthenticationError(f"Credential file not found: {file_path}")
        except Exception as exc:
            # Log the full error for debugging but raise a clean message
            self._logger.error(
                "manager_creation_failed",
                label=label,
                manager_class=manager_class_name,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise AuthenticationError(
                f"Failed to create manager from class '{manager_class_name}': {exc}"
            )

        self._logger.info(
            "provider_manager_created_from_class",
            manager_class=manager_class_name,
            storage_class=storage_class_name,
            custom_file=custom_file,
            label=label,
            manager_type=type(manager).__name__,
        )

        return manager  # type: ignore[no-any-return]

    def _import_class(self, class_path: str) -> type:
        """Dynamically import a class from a fully qualified path.

        Args:
            class_path: Fully qualified class path (e.g., 'module.submodule.ClassName')

        Returns:
            Imported class

        Raises:
            ValueError: If class path is invalid
            ImportError: If module cannot be imported
            AttributeError: If class not found in module
        """
        if "." not in class_path:
            raise ValueError(
                f"Invalid class path (must be fully qualified): {class_path}"
            )

        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if not isinstance(cls, type):
                raise ValueError(f"'{class_path}' is not a class")

            return cls
        except ImportError as exc:
            raise ImportError(f"Cannot import module '{module_path}': {exc}") from exc
        except AttributeError as exc:
            raise AttributeError(
                f"Module '{module_path}' has no class '{class_name}'"
            ) from exc

    async def _create_manager_with_custom_file(
        self,
        manager_key: str,
        file_path: str,
        label: str,
    ) -> AuthManager:
        """Create auth manager with custom file storage.

        Args:
            manager_key: Manager registry key
            file_path: Custom file path for storage
            label: Label for logging

        Returns:
            AuthManager instance with custom storage

        Raises:
            AuthenticationError: If manager class not found or creation fails
        """
        if self._auth_registry is None:
            raise AuthenticationError("Auth registry not available")

        # Get manager class from registry
        manager_class = self._auth_registry.get_class(manager_key)
        if manager_class is None:
            raise AuthenticationError(
                f"Manager class not found for key: {manager_key}. "
                "Only managers registered via register_class support custom file paths."
            )

        self._logger.debug(
            "creating_manager_with_custom_storage",
            manager_key=manager_key,
            file_path=file_path,
            label=label,
            manager_class=manager_class.__name__,
        )

        # Create custom storage based on manager type
        # file_path is already expanded and resolved by config validator
        custom_path = Path(file_path)
        storage = await self._create_storage_for_manager(
            manager_key, manager_class, custom_path
        )

        # Create manager with custom storage
        if hasattr(manager_class, "create"):
            manager = await manager_class.create(storage=storage)
        else:
            raise AuthenticationError(
                f"Manager class {manager_class.__name__} does not support async creation"
            )

        self._logger.info(
            "provider_manager_created_with_custom_storage",
            manager_key=manager_key,
            file_path=str(custom_path),
            label=label,
            manager_type=type(manager).__name__,
        )

        return manager  # type: ignore[no-any-return]

    async def _create_storage_for_manager(
        self,
        manager_key: str,
        manager_class: type,
        storage_path: Path,
    ) -> Any:
        """Create appropriate storage instance for the manager type.

        Args:
            manager_key: Manager registry key
            manager_class: Manager class
            storage_path: Path to storage file

        Returns:
            Storage instance

        Raises:
            AuthenticationError: If storage type cannot be determined
        """
        # Map manager keys to their storage classes
        # This could be made more dynamic by having managers expose their storage class
        if manager_key == "codex":
            from ccproxy.plugins.oauth_codex.storage import CodexTokenStorage

            return CodexTokenStorage(storage_path=storage_path)
        else:
            raise AuthenticationError(
                f"Custom file storage not yet supported for manager: {manager_key}. "
                f"Supported managers: codex. "
                f"Either use type='file' or add storage mapping for {manager_key}."
            )


__all__ = ["AuthManagerFactory"]
