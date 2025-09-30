"""Auth manager registry for plugin system."""

from collections.abc import Awaitable, Callable
from typing import Any

from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class AuthManagerRegistry:
    """Registry for auth managers that can be referenced by name.

    This registry uses Any types to avoid circular imports with the auth module.
    The actual auth managers are expected to have a 'create' class method for
    async instantiation.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Awaitable[Any]]] = {}
        self._instances: dict[str, Any] = {}

    def register_class(self, name: str, auth_manager_class: type[Any]) -> None:
        """Register an auth manager class that will be instantiated on demand.

        Args:
            name: Name to register the auth manager under
            auth_manager_class: Auth manager class to instantiate
        """

        async def create_instance() -> Any:
            # Use the async create class method if available
            if hasattr(auth_manager_class, "create"):
                instance = await auth_manager_class.create()
                return instance
            else:
                # Fallback to direct instantiation (requires storage parameter)
                raise RuntimeError(
                    f"Auth manager class {auth_manager_class.__name__} must have a 'create' class method"
                )

        self._factories[name] = create_instance
        logger.debug(
            "auth_manager_class_registered",
            name=name,
            class_name=auth_manager_class.__name__,
        )

    def register_factory(
        self, name: str, factory: Callable[[], Awaitable[Any]]
    ) -> None:
        """Register a factory function for creating auth managers.

        Args:
            name: Name to register the auth manager under
            factory: Factory function that returns an auth manager instance
        """
        self._factories[name] = factory
        logger.debug("auth_manager_factory_registered", name=name)

    def register_instance(self, name: str, instance: Any) -> None:
        """Register an existing auth manager instance.

        Args:
            name: Name to register the auth manager under
            instance: Auth manager instance
        """
        self._instances[name] = instance
        logger.debug(
            "auth_manager_instance_registered",
            name=name,
            instance_type=type(instance).__name__,
        )

    def unregister(self, name: str) -> None:
        """Unregister an auth manager.

        Args:
            name: Name of auth manager to unregister
        """
        if name in self._factories:
            del self._factories[name]
        if name in self._instances:
            del self._instances[name]
        logger.debug("auth_manager_unregistered", name=name)

    async def get(self, name: str) -> Any | None:
        """Get an auth manager by name.

        Args:
            name: Name of the auth manager to retrieve

        Returns:
            Auth manager instance or None if not found
        """
        # Check for existing instance first
        if name in self._instances:
            return self._instances[name]

        # Check for factory
        if name in self._factories:
            try:
                instance = await self._factories[name]()
                # Cache the instance for future use
                self._instances[name] = instance
                logger.debug(
                    "auth_manager_created",
                    name=name,
                    instance_type=type(instance).__name__,
                )
                return instance
            except Exception as e:
                logger.error(
                    "auth_manager_creation_failed", name=name, error=str(e), exc_info=e
                )
                return None

        return None

    def has(self, name: str) -> bool:
        """Check if an auth manager is registered under the given name.

        Args:
            name: Name to check

        Returns:
            True if auth manager is registered, False otherwise
        """
        return name in self._factories or name in self._instances

    def list(self) -> dict[str, str]:
        """List all registered auth managers.

        Returns:
            Dictionary mapping auth manager names to their types
        """
        result = {}

        # Add factories
        for name in self._factories:
            result[name] = "factory"

        # Add instances
        for name, instance in self._instances.items():
            result[name] = type(instance).__name__

        return result

    def get_class(self, name: str) -> type[Any] | None:
        """Get the auth manager class if registered via register_class.

        Args:
            name: Name of the auth manager

        Returns:
            Auth manager class or None if not found or not registered as class
        """
        # This is a bit hacky but works: we inspect the factory closure
        # to extract the class that was registered
        if name not in self._factories:
            return None

        factory = self._factories[name]
        # Check if this factory was created by register_class
        if hasattr(factory, "__closure__") and factory.__closure__:
            for cell in factory.__closure__:
                try:
                    obj = cell.cell_contents
                    if isinstance(obj, type):
                        return obj
                except (AttributeError, ValueError):
                    continue
        return None

    def clear(self) -> None:
        """Clear all registered auth managers."""
        self._factories.clear()
        self._instances.clear()
        logger.debug("auth_manager_registry_cleared")
