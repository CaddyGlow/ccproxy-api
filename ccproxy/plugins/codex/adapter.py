import json
import uuid
import httpx
from starlette.responses import Response

from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import filter_request_headers, extract_response_headers, to_canonical_headers
from ccproxy.core.logging import get_plugin_logger

from .models import CodexAuthData
from .detection_service import CodexDetectionService

logger = get_plugin_logger()

class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""
    
    def __init__(self, detection_service: CodexDetectionService, **kwargs):
        super().__init__(**kwargs) 
        self.detection_service = detection_service
    
    async def get_target_url(self, endpoint: str) -> str:
        return "https://chat.openai.com/backend-anon/responses"
    
    async def prepare_provider_request(
        self, 
        body: bytes, 
        headers: dict[str, str], 
        endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        
        # Get auth
        auth_data = await self.auth_manager.get_credentials()
        
        # Parse and convert body
        body_data = json.loads(body.decode()) if body else {}
        
        # Format conversion
        if self._needs_format_conversion(endpoint):
            body_data = await self._convert_to_codex_format(body_data)
        
        # Inject instructions
        if "instructions" not in body_data:
            body_data["instructions"] = self._get_instructions()
        
        if "stream" not in body_data:
            body_data["stream"] = True
        
        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        filtered_headers.update({
            "authorization": f"Bearer {auth_data.access_token}",
            "chatgpt-account-id": auth_data.chatgpt_account_id, 
            "session-id": str(uuid.uuid4()),
            "content-type": "application/json"
        })
        
        # Add CLI headers
        if self.detection_service:
            cli_headers = self.detection_service.get_headers()
            for key, value in cli_headers.items():
                filtered_headers[key.lower()] = value
        
        return json.dumps(body_data).encode(), filtered_headers
    
    async def process_provider_response(
        self, 
        response: httpx.Response, 
        endpoint: str
    ) -> Response:
        
        # Convert response format
        response_data = json.loads(response.content)
        converted_data = await self._convert_codex_to_openai(response_data)
        
        response_headers = extract_response_headers(response)
        
        return Response(
            content=json.dumps(converted_data).encode(),
            status_code=response.status_code,
            headers=to_canonical_headers(response_headers)
        )
    
    # Helper methods (move from transformers)
    def _needs_format_conversion(self, endpoint):
        return True  # Codex always needs conversion
    
    def _get_instructions(self):
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field
        return "You are a coding agent..."

    async def _convert_to_codex_format(self, body_data):
        # Use format adapter registry or inline conversion
        pass

    async def _convert_codex_to_openai(self, response_data):
        # Use format adapter registry or inline conversion  
        pass