"""HTTP-level transformers for proxy service."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class HTTPRequestTransformer:
    """Basic HTTP request transformer for proxy service."""
    
    def transform_path(self, path: str, proxy_mode: str = "full") -> str:
        """Transform request path."""
        # Basic path transformation - pass through for now
        return path
    
    def create_proxy_headers(
        self, 
        headers: dict[str, str], 
        access_token: str, 
        proxy_mode: str = "full"
    ) -> dict[str, str]:
        """Create proxy headers from original headers."""
        proxy_headers = {}
        
        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["host", "authorization"]:
                proxy_headers[key] = value
        
        # Add authentication
        proxy_headers["Authorization"] = f"Bearer {access_token}"
        
        # Set content type if not present
        if "content-type" not in proxy_headers:
            proxy_headers["Content-Type"] = "application/json"
        
        return proxy_headers
    
    def transform_request_body(
        self, 
        body: bytes, 
        path: str, 
        proxy_mode: str = "full"
    ) -> bytes:
        """Transform request body."""
        # Basic body transformation - pass through for now
        return body


class HTTPResponseTransformer:
    """Basic HTTP response transformer for proxy service."""
    
    def transform_response_body(
        self, 
        body: bytes, 
        path: str, 
        proxy_mode: str = "full"
    ) -> bytes:
        """Transform response body."""
        # Basic body transformation - pass through for now
        return body
    
    def transform_response_headers(
        self, 
        headers: dict[str, str], 
        path: str, 
        content_length: int, 
        proxy_mode: str = "full"
    ) -> dict[str, str]:
        """Transform response headers."""
        transformed_headers = {}
        
        # Copy important headers
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key not in ["content-length", "transfer-encoding"]:
                transformed_headers[key] = value
        
        # Set content length
        transformed_headers["Content-Length"] = str(content_length)
        
        # Add CORS headers
        transformed_headers["Access-Control-Allow-Origin"] = "*"
        transformed_headers["Access-Control-Allow-Headers"] = "*"
        transformed_headers["Access-Control-Allow-Methods"] = "*"
        
        return transformed_headers
    
    def _is_openai_request(self, path: str) -> bool:
        """Check if this is an OpenAI API request."""
        return "/openai/" in path or "/chat/completions" in path