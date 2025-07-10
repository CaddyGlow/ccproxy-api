"""
FastAPI middleware for automatic metrics collection.

This module provides middleware to automatically collect metrics for
all requests processed by the FastAPI application.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .collector import MetricsCollector

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic metrics collection.
    
    This middleware automatically collects request and response metrics
    for all API endpoints.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        collector: MetricsCollector,
        capture_request_body: bool = False,
        capture_response_body: bool = False,
        excluded_paths: Optional[list[str]] = None,
        sample_rate: float = 1.0
    ):
        """
        Initialize the metrics middleware.
        
        Args:
            app: FastAPI application
            collector: Metrics collector instance
            capture_request_body: Whether to capture request body content
            capture_response_body: Whether to capture response body content
            excluded_paths: List of paths to exclude from metrics
            sample_rate: Sampling rate for metrics (0.0-1.0)
        """
        super().__init__(app)
        self.collector = collector
        self.capture_request_body = capture_request_body
        self.capture_response_body = capture_response_body
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.sample_rate = sample_rate
        
        # Internal state
        self._request_contexts: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler
            
        Returns:
            Response object
        """
        # Check if path should be excluded
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Check sampling rate
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                return await call_next(request)
        
        # Generate unique request ID
        request_id = str(uuid4())
        
        # Store request context
        start_time = time.time()
        request_context = {
            "request_id": request_id,
            "start_time": start_time,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
        }
        
        # Extract user information if available
        user_id = self._extract_user_id(request)
        session_id = self._extract_session_id(request)
        
        # Collect request start metrics
        try:
            await self.collector.collect_request_start(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                endpoint=self._get_endpoint_name(request),
                api_version=self._get_api_version(request),
                user_id=user_id,
                session_id=session_id,
                client_ip=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                content_length=request.headers.get("content-length"),
                content_type=request.headers.get("content-type"),
                model=self._extract_model(request),
                provider=self._extract_provider(request),
                streaming=self._is_streaming_request(request),
                **self._extract_request_params(request)
            )
        except Exception as e:
            logger.error(f"Failed to collect request start metrics: {e}")
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            # Create error response
            response = Response(
                content=f"Internal server error: {str(e)}",
                status_code=500,
                media_type="text/plain"
            )
        
        # Calculate response time
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Collect response metrics
        try:
            await self.collector.collect_response(
                request_id=request_id,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                content_length=self._get_content_length(response),
                content_type=response.headers.get("content-type"),
                **self._extract_response_tokens(response)
            )
        except Exception as e:
            logger.error(f"Failed to collect response metrics: {e}")
        
        # Collect error metrics if there was an error
        if error:
            try:
                await self.collector.collect_error(
                    request_id=request_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    endpoint=self._get_endpoint_name(request),
                    method=request.method,
                    status_code=response.status_code
                )
            except Exception as e:
                logger.error(f"Failed to collect error metrics: {e}")
        
        # Collect latency metrics
        try:
            await self.collector.collect_latency(
                request_id=request_id,
                total_latency_ms=response_time_ms,
                request_processing_ms=response_time_ms,  # Could be more granular
            )
        except Exception as e:
            logger.error(f"Failed to collect latency metrics: {e}")
        
        # Clean up request context
        try:
            await self.collector.finish_request(request_id)
        except Exception as e:
            logger.error(f"Failed to finish request context: {e}")
        
        return response
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if a path should be excluded from metrics."""
        return any(path.startswith(excluded) for excluded in self.excluded_paths)
    
    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request."""
        # Check for user in request state (set by auth middleware)
        if hasattr(request.state, "user"):
            user = request.state.user
            if hasattr(user, "id"):
                return str(user.id)
            elif hasattr(user, "user_id"):
                return str(user.user_id)
        
        # Check for user ID in headers
        user_id = request.headers.get("x-user-id")
        if user_id:
            return user_id
        
        # Check for bearer token and extract user info
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Could decode JWT token here to extract user info
            # For now, just return None
            pass
        
        return None
    
    def _extract_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request."""
        # Check for session in request state
        if hasattr(request.state, "session_id"):
            return request.state.session_id
        
        # Check for session ID in headers
        session_id = request.headers.get("x-session-id")
        if session_id:
            return session_id
        
        # Check for session cookie
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            return session_cookie
        
        return None
    
    def _get_endpoint_name(self, request: Request) -> str:
        """Get the endpoint name for the request."""
        # Try to get from route
        if hasattr(request, "route") and request.route:
            if hasattr(request.route, "path"):
                return request.route.path
            elif hasattr(request.route, "name"):
                return request.route.name
        
        # Fall back to path
        return request.url.path
    
    def _get_api_version(self, request: Request) -> str:
        """Get API version from request."""
        # Check for version in path
        path = request.url.path
        if path.startswith("/v1/"):
            return "v1"
        elif path.startswith("/openai/v1/"):
            return "openai-v1"
        
        # Check for version in headers
        version = request.headers.get("x-api-version")
        if version:
            return version
        
        # Default version
        return "v1"
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return None
    
    def _extract_model(self, request: Request) -> Optional[str]:
        """Extract model name from request."""
        # This would need to be implemented based on request body parsing
        # For now, return None - could be enhanced to parse JSON body
        return None
    
    def _extract_provider(self, request: Request) -> Optional[str]:
        """Extract provider from request path."""
        path = request.url.path
        if path.startswith("/openai/"):
            return "openai"
        elif path.startswith("/v1/"):
            return "anthropic"
        
        return None
    
    def _is_streaming_request(self, request: Request) -> bool:
        """Check if request is for streaming response."""
        # This would need to be implemented based on request body parsing
        # For now, return False - could be enhanced to parse JSON body
        return False
    
    def _extract_request_params(self, request: Request) -> Dict[str, Any]:
        """Extract request parameters for metrics."""
        params = {}
        
        # Add query parameters
        if request.query_params:
            params["query_params"] = dict(request.query_params)
        
        # Could add parsed body parameters here
        # For now, return empty dict
        return params
    
    def _get_content_length(self, response: Response) -> Optional[int]:
        """Get content length from response."""
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        # Try to get from body if available
        if hasattr(response, "body") and response.body:
            return len(response.body)
        
        return None
    
    def _extract_response_tokens(self, response: Response) -> Dict[str, Any]:
        """Extract token information from response."""
        tokens = {}
        
        # Try to parse from response headers
        input_tokens = response.headers.get("x-input-tokens")
        if input_tokens:
            try:
                tokens["input_tokens"] = int(input_tokens)
            except ValueError:
                pass
        
        output_tokens = response.headers.get("x-output-tokens")
        if output_tokens:
            try:
                tokens["output_tokens"] = int(output_tokens)
            except ValueError:
                pass
        
        cache_read_tokens = response.headers.get("x-cache-read-tokens")
        if cache_read_tokens:
            try:
                tokens["cache_read_tokens"] = int(cache_read_tokens)
            except ValueError:
                pass
        
        cache_write_tokens = response.headers.get("x-cache-write-tokens")
        if cache_write_tokens:
            try:
                tokens["cache_write_tokens"] = int(cache_write_tokens)
            except ValueError:
                pass
        
        # Could parse from response body JSON here
        # For now, return what we found in headers
        return tokens


class AsyncMetricsMiddleware:
    """
    Alternative async metrics middleware for more advanced use cases.
    """
    
    def __init__(
        self,
        collector: MetricsCollector,
        **kwargs: Any
    ):
        """
        Initialize the async metrics middleware.
        
        Args:
            collector: Metrics collector instance
            **kwargs: Additional configuration options
        """
        self.collector = collector
        self.config = kwargs
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with async metrics collection.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler
            
        Returns:
            Response object
        """
        # Generate unique request ID
        request_id = str(uuid4())
        
        # Use the collector's request context manager
        async with self.collector.request_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            endpoint=request.url.path,
            api_version="v1"
        ) as request_metric:
            
            # Process request
            try:
                response = await call_next(request)
                
                # Collect response metrics
                await self.collector.collect_response(
                    request_id=request_id,
                    status_code=response.status_code
                )
                
                return response
                
            except Exception as e:
                # Collect error metrics
                await self.collector.collect_error(
                    request_id=request_id,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                
                raise