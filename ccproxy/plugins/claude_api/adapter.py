import json
from typing import Any
import httpx
from starlette.responses import Response, StreamingResponse

from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import filter_request_headers, extract_response_headers, to_canonical_headers
from ccproxy.core.logging import get_plugin_logger

from .models import ClaudeAPIAuthData
from .detection_service import ClaudeAPIDetectionService

logger = get_plugin_logger()

class ClaudeAPIAdapter(BaseHTTPAdapter):
    """Simplified Claude API adapter."""
    
    def __init__(self, detection_service: ClaudeAPIDetectionService, **kwargs):
        super().__init__(**kwargs)
        self.detection_service = detection_service
    
    async def get_target_url(self, endpoint: str) -> str:
        return "https://api.anthropic.com/v1/messages"
    
    async def prepare_provider_request(
        self, 
        body: bytes, 
        headers: dict[str, str], 
        endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        
        # Get auth
        auth_data = await self.auth_manager.get_credentials()
        access_token = auth_data.access_token
        
        # Parse body
        body_data = json.loads(body.decode()) if body else {}
        
        # Inject system prompt if available
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.system_prompt:
                body_data = self._inject_system_prompt(body_data, cached_data.system_prompt)
        
        # Format conversion if needed
        if self._needs_openai_conversion(endpoint):
            body_data = await self._convert_openai_to_anthropic(body_data)
        
        # Filter headers  
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        filtered_headers["authorization"] = f"Bearer {access_token}"
        
        # Add CLI headers if available
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data() 
            if cached_data and cached_data.headers:
                cli_headers = cached_data.headers.to_headers_dict()
                for key, value in cli_headers.items():
                    filtered_headers[key.lower()] = value
        
        return json.dumps(body_data).encode(), filtered_headers
    
    async def process_provider_response(
        self, 
        response: httpx.Response, 
        endpoint: str
    ) -> Response:
        
        response_headers = extract_response_headers(response)
        content = response.content
        
        # Format conversion if needed
        if self._needs_anthropic_conversion(endpoint):
            response_data = json.loads(content)
            converted_data = await self._convert_anthropic_to_openai(response_data)
            content = json.dumps(converted_data).encode()
        
        return Response(
            content=content,
            status_code=response.status_code,
            headers=to_canonical_headers(response_headers)
        )
    
    # Helper methods (move from transformers)
    def _inject_system_prompt(self, body_data, system_prompt):
        # Move logic from transformer here
        pass
    
    def _needs_openai_conversion(self, endpoint):
        return endpoint.endswith("/chat/completions")
    
    async def _convert_openai_to_anthropic(self, body_data):
        # Use format adapter registry or inline conversion
        pass

    def _needs_anthropic_conversion(self, endpoint):
        return endpoint.endswith("/chat/completions")
    
    async def _convert_anthropic_to_openai(self, response_data):
        # Use format adapter registry or inline conversion
        pass