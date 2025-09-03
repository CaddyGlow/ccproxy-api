"""Base credentials protocol for all authentication implementations."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseCredentials(Protocol):
    """Protocol that all credential implementations must follow.

    This defines the contract for credentials without depending on
    any specific provider implementation.
    """

    def is_expired(self) -> bool:
        """Check if the credentials are expired.

        Returns:
            True if expired, False otherwise
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Convert credentials to dictionary for storage.

        Returns:
            Dictionary representation of credentials
        """
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseCredentials":
        """Create credentials from dictionary.

        Args:
            data: Dictionary containing credential data

        Returns:
            Credentials instance
        """
        ...
