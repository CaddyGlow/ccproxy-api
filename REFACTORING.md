```
ccproxy/
├── __init__.py
├── __main__.py                    # Entry point: python -m ccproxy
├── __version__.py                 # Single source of version truth
│
├── core/                          # Core proxy functionality
│   ├── __init__.py
│   ├── proxy.py                   # Base proxy service
│   ├── transformers.py            # Abstract transformer interfaces
│   ├── middleware.py              # Request/response middleware
│   ├── errors.py                  # Centralized error definitions
│   └── types.py                   # Core type definitions
│
├── adapters/                      # API format adapters
│   ├── __init__.py
│   ├── base.py                    # Abstract adapter interface
│   ├── anthropic/
│   │   ├── __init__.py
│   │   ├── adapter.py             # Anthropic format adapter
│   │   ├── models.py              # Anthropic-specific models
│   │   └── streaming.py           # Anthropic SSE handling
│   └── openai/
│       ├── __init__.py
│       ├── adapter.py             # OpenAI format adapter
│       ├── models.py              # OpenAI-specific models
│       └── streaming.py           # OpenAI SSE handling
│
├── auth/                          # Authentication module
│   ├── __init__.py
│   ├── manager.py                 # Main auth manager
│   ├── oauth/
│   │   ├── __init__.py
│   │   ├── client.py              # OAuth client implementation
│   │   ├── models.py              # OAuth-specific models
│   │   └── storage.py             # Token storage interface
│   ├── bearer.py                  # Bearer token auth
│   └── dependencies.py            # FastAPI auth dependencies
│
├── services/                      # Business logic services
│   ├── __init__.py
│   ├── proxy_service.py           # Pure reverse proxy
│   ├── claude_sdk_service.py      # Claude SDK integration
│   ├── hybrid_service.py          # Combined proxy + SDK
│   └── transformer_service.py     # Request/response transformation
│
├── claude_sdk/                    # Claude Code SDK integration
│   ├── __init__.py
│   ├── client.py                  # Minimal SDK wrapper
│   ├── converter.py               # Message format converter
│   └── options.py                 # SDK options handling
│
├── api/                           # FastAPI application
│   ├── __init__.py
│   ├── app.py                     # FastAPI app factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── proxy.py               # Proxy endpoints
│   │   ├── claude.py              # Claude SDK endpoints
│   │   ├── health.py              # Health check endpoints
│   │   └── auth.py                # Auth endpoints
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py                # Authentication middleware
│   │   ├── cors.py                # CORS middleware
│   │   ├── logging.py             # Request logging
│   │   └── errors.py              # Error handling middleware
│   └── dependencies.py            # Shared FastAPI dependencies
│
├── config/                        # Configuration management
│   ├── __init__.py
│   ├── settings.py                # Pydantic settings classes
│   ├── loader.py                  # Config file loader
│   └── validators.py              # Config validation
│
├── cli/                           # CLI interface
│   ├── __init__.py
│   ├── main.py                    # Main CLI app
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── serve.py               # Server commands
│   │   ├── auth.py                # Auth management commands
│   │   ├── config.py              # Config commands
│   │   └── utils.py               # Utility commands
│   └── helpers.py                 # CLI utilities
│
├── docker/                        # Docker functionality (keep existing)
│   ├── __init__.py
│   ├── adapter.py                 # Docker adapter
│   ├── builder.py                 # Command builder
│   ├── models.py                  # Docker-specific models
│   └── validators.py              # Docker param validation
│
├── utils/                         # Shared utilities
│   ├── __init__.py
│   ├── logging.py                 # Logging configuration
│   ├── async_helpers.py           # Async utilities
│   ├── validators.py              # General validators
│   └── constants.py               # Shared constants
│
└── tests/                         # Test directory structure
    ├── __init__.py
    ├── conftest.py                # Pytest fixtures
    ├── unit/                      # Unit tests
    │   ├── __init__.py
    │   ├── core/
    │   ├── adapters/
    │   ├── auth/
    │   └── services/
    ├── integration/               # Integration tests
    │   ├── __init__.py
    │   ├── test_proxy_flow.py
    │   ├── test_claude_sdk_flow.py
    │   └── test_auth_flow.py
    └── e2e/                       # End-to-end tests
        ├── __init__.py
        └── test_full_scenarios.py

# Configuration files at root
.env.example                       # Example environment variables
config.example.toml                # Example TOML config
pyproject.toml                     # Project metadata and dependencies
requirements.txt                   # Production dependencies
requirements-dev.txt               # Development dependencies
Dockerfile                         # Container definition
docker-compose.yml                 # Local development setup
```

## Detailed Module Descriptions:

### Core Module (`core/`)
```python
# core/proxy.py
class ProxyClient:
    """Base HTTP client for proxying requests"""
    def __init__(self, base_url: str, timeout: float = 120.0):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def forward(self, request: Request) -> Response:
        """Forward request to target"""

# core/transformers.py
class TransformerPipeline:
    """Chain multiple transformers"""
    def __init__(self, transformers: List[Transformer]):
        self.transformers = transformers

    async def transform_request(self, request: Dict) -> Dict:
        for transformer in self.transformers:
            request = await transformer.transform_request(request)
        return request
```

### Adapters Module (`adapters/`)
```python
# adapters/base.py
class APIAdapter(ABC):
    """Base adapter for API format conversion"""

    @abstractmethod
    async def adapt_request(self, request: Dict) -> Dict:
        pass

    @abstractmethod
    async def adapt_response(self, response: Dict) -> Dict:
        pass

    @abstractmethod
    async def adapt_stream(self, stream: AsyncIterator) -> AsyncIterator:
        pass

# adapters/openai/adapter.py
class OpenAIAdapter(APIAdapter):
    """Convert between OpenAI and Anthropic formats"""

    def __init__(self):
        self.request_converter = OpenAIRequestConverter()
        self.response_converter = OpenAIResponseConverter()
        self.stream_converter = OpenAIStreamConverter()
```

### Services Module (`services/`)
```python
# services/proxy_service.py
class ProxyService:
    """Pure reverse proxy service"""

    def __init__(
        self,
        proxy_client: ProxyClient,
        auth_manager: AuthManager,
        transformer_pipeline: TransformerPipeline
    ):
        self.proxy_client = proxy_client
        self.auth_manager = auth_manager
        self.transformer_pipeline = transformer_pipeline

    async def handle_request(
        self,
        request: Request,
        path: str,
        user: Optional[User] = None
    ) -> Response:
        """Process and forward request"""

# services/claude_sdk_service.py
class ClaudeSDKService:
    """Handle Claude SDK operations"""

    def __init__(self, sdk_client: ClaudeSDKClient):
        self.sdk_client = sdk_client

    async def execute(
        self,
        messages: List[Dict],
        options: Dict
    ) -> Union[Dict, AsyncIterator[Dict]]:
        """Execute Claude SDK query"""
```

### API Module (`api/`)
```python
# api/app.py
def create_app(config: Settings) -> FastAPI:
    """Factory function to create FastAPI app"""

    app = FastAPI(
        title="Claude Code Proxy",
        version=__version__,
        docs_url="/docs" if config.enable_docs else None
    )

    # Register middleware
    app.add_middleware(AuthMiddleware, auth_manager=get_auth_manager())
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)

    # Register routes
    app.include_router(proxy_router, prefix="/api")
    app.include_router(claude_router, prefix="/claude")
    app.include_router(health_router, prefix="/health")

    return app

# api/routes/proxy.py
@router.post("/v1/messages")
async def messages_endpoint(
    request: Request,
    body: Dict = Body(...),
    service: ProxyService = Depends(get_proxy_service),
    user: User = Depends(get_current_user)
):
    """Handle messages endpoint"""
    return await service.handle_request(request, "/v1/messages", user)
```

### Config Module (`config/`)
```python
# config/settings.py
class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Server
    host: str = Field(default="127.0.0.1", env="CCPROXY_HOST")
    port: int = Field(default=8000, env="CCPROXY_PORT")

    # Proxy
    proxy_mode: ProxyMode = Field(default="full", env="CCPROXY_MODE")
    target_url: str = Field(default="https://api.anthropic.com", env="CCPROXY_TARGET")

    # Auth
    auth_enabled: bool = Field(default=True, env="CCPROXY_AUTH_ENABLED")
    auth_token: Optional[str] = Field(default=None, env="CCPROXY_AUTH_TOKEN")

    # Features
    enable_claude_sdk: bool = Field(default=True, env="CCPROXY_ENABLE_SDK")
    enable_openai_compat: bool = Field(default=True, env="CCPROXY_ENABLE_OPENAI")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# config/loader.py
class ConfigLoader:
    """Load configuration from multiple sources"""

    def load(self) -> Settings:
        """Load from env vars, .env file, and config files"""
        # Priority: ENV > .env > config.toml > defaults
```

### CLI Module (`cli/`)
```python
# cli/main.py
app = typer.Typer(
    name="ccproxy",
    help="Claude Code Proxy - API gateway for Claude"
)

@app.command()
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    mode: str = typer.Option("proxy", "--mode", "-m"),
    config: Optional[Path] = typer.Option(None, "--config", "-c")
):
    """Start the proxy server"""
    settings = load_settings(config)
    run_server(settings, host, port, mode)

@app.command()
def auth(ctx: typer.Context):
    """Authentication management"""
    # Sub-commands for login, logout, status

@app.command()
def config(ctx: typer.Context):
    """Configuration management"""
    # Sub-commands for show, validate, init
```

## Key Design Principles:

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Services receive dependencies through constructors
3. **Interface Segregation**: Small, focused interfaces rather than large ones
4. **Layered Architecture**: Clear separation between API, services, and core
5. **Testability**: All components can be easily mocked and tested
6. **Configuration**: Centralized configuration with clear precedence
7. **Extensibility**: Easy to add new adapters, services, or middleware

This structure makes the codebase:
- **Easier to navigate**: Clear organization by functionality
- **Easier to test**: Isolated components with clear boundaries
- **Easier to maintain**: Changes are localized to specific modules
- **Easier to extend**: New features have obvious homes
- **Easier to deploy**: Can be packaged as a single module or microservices
