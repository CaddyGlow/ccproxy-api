"""
Application bootstrapping and dependency injection container setup.

This module is responsible for the initial setup of the application's core services,
including configuration loading and service container initialization. It acts as the
main entry point for assembling the application's components before the web server
starts.
"""

from ccproxy.config.settings import Settings
from ccproxy.services.container import ServiceContainer


def create_service_container(settings: Settings | None = None) -> ServiceContainer:
    """
    Create and configure the service container.

    Args:
        settings: Optional pre-loaded settings instance. If not provided,
                  settings will be loaded from config files/environment.

    Returns:
        The initialized service container.
    """
    if settings is None:
        settings = Settings.from_config()

    container = ServiceContainer(settings)

    # You can add core, non-plugin service registrations here if needed.
    # For example:
    # from ccproxy.services.some_service import SomeService
    # container.register_service(SomeService)

    return container
