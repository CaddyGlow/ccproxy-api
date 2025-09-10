# Placeholder processor for backward compatibility during migration
# This is a temporary stub to fix import errors

class RequestProcessor:
    def __init__(self, logger=None):
        self.logger = logger
    
    async def process_request(self, body, headers, handler_config, **kwargs):
        # Simple pass-through processing
        is_streaming = "stream" in str(body) and "true" in str(body)
        return body, headers, is_streaming
    
    async def process_response(self, response, headers, handler_config, **kwargs):
        # Simple pass-through processing
        return response, headers