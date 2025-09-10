import json
import uuid
from typing import Any
import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import filter_request_headers, extract_response_headers, to_canonical_headers
from ccproxy.core.logging import get_plugin_logger

from .config import CopilotConfig
from .models import CopilotAuthData
from .oauth.provider import CopilotOAuthProvider

logger = get_plugin_logger()

class CopilotAdapter(BaseHTTPAdapter):
    """Simplified Copilot adapter."""
    
    def __init__(self, oauth_provider: CopilotOAuthProvider, config: CopilotConfig, **kwargs):
        super().__init__(**kwargs)
        self.oauth_provider = oauth_provider
        self.config = config
    
    async def get_target_url(self, endpoint: str) -> str:
        return "https://api.githubcopilot.com/chat/completions"
    
    async def prepare_provider_request(
        self, 
        body: bytes, 
        headers: dict[str, str], 
        endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        
        # Get auth token
        access_token = await self.oauth_provider.ensure_copilot_token()
        
        # Filter headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        
        # Add Copilot headers (lowercase keys)
        copilot_headers = {}
        for key, value in self.config.api_headers.items():
            copilot_headers[key.lower()] = value
        
        copilot_headers["authorization"] = f"Bearer {access_token}"  
        copilot_headers["x-request-id"] = str(uuid.uuid4())
        
        # Merge headers
        final_headers = {}
        final_headers.update(filtered_headers) 
        final_headers.update(copilot_headers)
        
        logger.debug("copilot_request_prepared", header_count=len(final_headers))
        
        return body, final_headers
    
    async def process_provider_response(
        self, 
        response: httpx.Response, 
        endpoint: str
    ) -> Response:
        
        response_headers = extract_response_headers(response)
        
        # Filter response headers
        safe_headers = {}
        for key, value in response_headers.items():
            if key not in {"connection", "transfer-encoding", "content-encoding"}:
                safe_headers[key] = value
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=to_canonical_headers(safe_headers)
        )