"""Main FastAPI application for Claude Proxy API Server."""

# Import from the new API structure
from ccproxy.api.app import create_app


# For backward compatibility, create app instance
app = create_app()
