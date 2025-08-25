"""Protocol interfaces for dependency inversion."""

from typing import Protocol


class IRequestHandler(Protocol):
    """Protocol for request handling functionality.

    Note: The dispatch_request method has been removed in favor of
    using plugin adapters' handle_request() method directly.
    """

    pass
