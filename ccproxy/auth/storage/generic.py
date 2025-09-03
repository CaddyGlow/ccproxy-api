"""Generic storage implementation using Pydantic validation."""

from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import SecretStr, TypeAdapter

from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.auth.storage.base import BaseJsonStorage
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseCredentials)


class GenericJsonStorage(BaseJsonStorage[T]):
    """Generic storage implementation using Pydantic validation.

    This replaces provider-specific storage classes with a single
    implementation that handles any Pydantic model.
    """

    def __init__(self, file_path: Path, model_class: type[T]):
        """Initialize generic storage.

        Args:
            file_path: Path to JSON file
            model_class: Pydantic model class for validation
        """
        super().__init__(file_path)
        self.model_class = model_class
        self.type_adapter = TypeAdapter(model_class)

    async def load(self) -> T | None:
        """Load and validate credentials with Pydantic.

        Returns:
            Validated model instance or None if file doesn't exist
        """
        try:
            data = await self._read_json()
        except Exception as e:
            # Handle JSON decode errors and other file read issues
            logger.error(
                "Failed to read credentials file",
                error=str(e),
                path=str(self.file_path),
                category="auth",
            )
            return None

        if not data:
            return None

        try:
            # Pydantic handles all validation and conversion
            return self.type_adapter.validate_python(data)
        except Exception as e:
            logger.error(
                "Failed to validate credentials",
                error=str(e),
                model=self.model_class.__name__,
            )
            return None

    async def save(self, obj: T) -> bool:
        """Save model using Pydantic serialization.

        Args:
            obj: Pydantic model instance to save

        Returns:
            True if saved successfully
        """
        try:
            # Preserve original JSON structure using aliases
            # Use dump_python without mode="json" to get actual values
            data = self.type_adapter.dump_python(
                obj,
                by_alias=True,  # Use field aliases from original models
                exclude_none=True,
            )
            # Convert SecretStr values to their actual values
            data = self._unmask_secrets(data)
            await self._write_json(data)
            return True
        except Exception as e:
            logger.error(
                "Failed to save credentials",
                error=str(e),
                model=self.model_class.__name__,
            )
            return False

    def _unmask_secrets(self, data: Any) -> Any:
        """Recursively unmask SecretStr values in data structure.

        Args:
            data: Data structure potentially containing SecretStr values

        Returns:
            Data with SecretStr values replaced by their actual values
        """
        if isinstance(data, dict):
            return {k: self._unmask_secrets(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._unmask_secrets(item) for item in data]
        elif isinstance(data, SecretStr):
            return data.get_secret_value()
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data
