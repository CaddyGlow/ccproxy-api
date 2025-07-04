# Architecture Documentation

## System Overview

The Claude Code Proxy API Server is a high-performance FastAPI application that provides both Anthropic and OpenAI-compatible interfaces for Claude AI models. The system acts as a translation layer between standard AI API formats and the Claude Code SDK, enabling seamless integration with existing applications.

## Core Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Load Balancer │    │   Proxy Server  │
│                 │────│                 │────│                 │
│ • Web Apps      │    │ • nginx/HAProxy │    │ • FastAPI App   │
│ • Mobile Apps   │    │ • Rate Limiting │    │ • API Routes    │
│ • CLI Tools     │    │ • SSL/TLS       │    │ • Middleware    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │  Claude Client  │
                                              │                 │
                                              │ • Claude SDK    │
                                              │ • CLI Interface │
                                              │ • Auth Handler  │
                                              └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Claude AI API   │
                                              │                 │
                                              │ • Model Access  │
                                              │ • Authentication│
                                              │ • Rate Limits   │
                                              └─────────────────┘
```

### Component Responsibilities

#### 1. API Layer
- **Location**: `claude_code_proxy/api/`
- **Purpose**: Handles HTTP requests and responses
- **Components**:
  - `v1/chat.py`: Anthropic-compatible endpoints
  - `openai/chat.py`: OpenAI-compatible endpoints
  - `openai/models.py`: Model listing endpoints

#### 2. Service Layer
- **Location**: `claude_code_proxy/services/`
- **Purpose**: Business logic and external service integration
- **Components**:
  - `claude_client.py`: Claude SDK integration
  - `streaming.py`: Response streaming (Anthropic format)
  - `openai_streaming.py`: OpenAI format streaming
  - `translator.py`: Format translation between APIs

#### 3. Data Models
- **Location**: `claude_code_proxy/models/`
- **Purpose**: Data validation and serialization
- **Components**:
  - `requests.py`: Request schemas (Anthropic format)
  - `responses.py`: Response schemas (Anthropic format)
  - `openai_models.py`: OpenAI format schemas
  - `errors.py`: Error handling schemas

#### 4. Configuration Management
- **Location**: `claude_code_proxy/config/`
- **Purpose**: Application configuration and settings
- **Components**:
  - `settings.py`: Pydantic-based configuration
  - Docker settings management
  - Claude CLI path resolution

#### 5. Exception Handling
- **Location**: `claude_code_proxy/exceptions.py`
- **Purpose**: Custom exception definitions
- **Features**:
  - Typed error responses
  - HTTP status code mapping
  - Error logging integration

## Data Flow Architecture

### Request Processing Flow

```
HTTP Request
     │
     ▼
┌─────────────────┐
│  FastAPI App    │
│                 │
│ • CORS Handling │
│ • Request Logs  │
│ • Auth Check    │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Route Handler  │
│                 │
│ • Path Matching │
│ • Method Check  │
│ • Rate Limiting │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Request Models  │
│                 │
│ • Validation    │
│ • Serialization │
│ • Type Checking │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Format Translator│
│                 │
│ • OpenAI → SDK  │
│ • Anthropic → SDK│
│ • Parameter Map │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Claude Client  │
│                 │
│ • SDK Interface │
│ • Auth Handling │
│ • Error Mapping │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Response Format │
│                 │
│ • SDK → API     │
│ • Stream Handle │
│ • Error Format  │
└─────────────────┘
     │
     ▼
HTTP Response
```

### Streaming Response Flow

```
Client Request (stream=true)
     │
     ▼
┌─────────────────┐
│  Stream Handler │
│                 │
│ • SSE Headers   │
│ • Async Context │
│ • Error Handling│
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Claude SDK Call │
│                 │
│ • Streaming Mode│
│ • Async Iterator│
│ • Chunk Buffer  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Stream Formatter│
│                 │
│ • Chunk Split   │
│ • SSE Format    │
│ • Progress Track│
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Client Response │
│                 │
│ • Event Stream  │
│ • Real-time     │
│ • Error Recovery│
└─────────────────┘
```

## API Layer Architecture

### Endpoint Structure

```
/
├── /health                     # Health check endpoint
├── /v1/                       # Anthropic-compatible API
│   ├── chat/completions       # Chat completions
│   └── models                 # Model listing
└── /openai/v1/                # OpenAI-compatible API
    ├── chat/completions       # Chat completions (OpenAI format)
    └── models                 # Model listing (OpenAI format)
```

### Request/Response Handling

#### Anthropic Format (v1/)
- **Request**: Direct Anthropic API format
- **Response**: Anthropic API format
- **Streaming**: Server-Sent Events (SSE) with `data:` prefix

#### OpenAI Format (openai/v1/)
- **Request**: OpenAI API format
- **Response**: OpenAI API format
- **Streaming**: OpenAI streaming format with `data:` chunks

## Service Layer Design

### Claude Client Service

```python
class ClaudeClient:
    """Main service for Claude API interactions"""
    
    # Authentication handling
    # Request/response processing
    # Error handling and retries
    # Streaming support
    # Model management
```

**Key Features**:
- Automatic authentication via Claude CLI
- Request retry logic with exponential backoff
- Streaming response handling
- Error mapping to HTTP status codes
- Model validation and availability checking

### Streaming Services

#### Anthropic Streaming (`streaming.py`)
- Handles Claude SDK streaming responses
- Formats responses as Server-Sent Events
- Maintains connection state
- Implements error recovery

#### OpenAI Streaming (`openai_streaming.py`)
- Converts Claude responses to OpenAI format
- Handles OpenAI-specific streaming requirements
- Maintains compatibility with OpenAI clients
- Implements text chunking for streaming

### Translation Service

The translator service handles format conversion between OpenAI and Anthropic formats:

```python
class OpenAITranslator:
    """Translates between OpenAI and Anthropic formats"""
    
    # Request translation: OpenAI → Anthropic
    # Response translation: Anthropic → OpenAI
    # Parameter mapping
    # Error format conversion
```

## Configuration Architecture

### Settings Management

```python
class Settings(BaseSettings):
    """Pydantic-based configuration management"""
    
    # Server configuration
    # Claude CLI integration
    # Docker settings
    # Authentication settings
    # Logging configuration
```

### Configuration Sources (Priority Order)

1. **Environment Variables**: Runtime configuration
2. **Configuration Files**: `.env` files
3. **Default Values**: Built-in defaults
4. **Auto-detection**: Claude CLI path resolution

### Docker Integration

```python
class DockerSettings:
    """Docker-specific configuration"""
    
    # Container image settings
    # Volume mounts
    # Environment variables
    # Network configuration
```

## Error Handling Architecture

### Exception Hierarchy

```
ClaudeProxyError (Base)
├── ValidationError
├── AuthenticationError
├── PermissionError
├── NotFoundError
├── RateLimitError
├── ModelNotFoundError
├── TimeoutError
└── ServiceUnavailableError
```

### Error Response Format

#### Anthropic Format
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Description of the error"
  }
}
```

#### OpenAI Format
```json
{
  "error": {
    "message": "Description of the error",
    "type": "invalid_request_error",
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

## Security Architecture

### Authentication Flow

```
Client Request
     │
     ▼
┌─────────────────┐
│  API Gateway    │
│                 │
│ • Rate Limiting │
│ • CORS Headers  │
│ • Input Valid   │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Claude Auth     │
│                 │
│ • CLI Auth      │
│ • Token Refresh │
│ • Session Mgmt  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Request Process │
│                 │
│ • Validated Req │
│ • Authorized    │
│ • Rate Limited  │
└─────────────────┘
```

### Security Features

1. **Authentication**: Claude CLI-based authentication
2. **Rate Limiting**: Built-in request rate limiting
3. **Input Validation**: Pydantic-based request validation
4. **CORS Handling**: Configurable cross-origin support
5. **Error Sanitization**: Secure error message handling

## Performance Architecture

### Optimization Strategies

1. **Async Processing**: Full async/await support
2. **Connection Pooling**: HTTP connection reuse
3. **Streaming Responses**: Real-time response streaming
4. **Request Caching**: Configurable response caching
5. **Lightweight Framework**: FastAPI for high performance

### Monitoring Points

1. **Request Metrics**: Response times, request counts
2. **Error Rates**: Error frequency and types
3. **Stream Health**: Streaming connection stability
4. **Resource Usage**: Memory and CPU utilization
5. **Claude API Health**: Upstream service availability

## Deployment Architecture

### Production Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Proxy Server  │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • nginx/HAProxy │    │ • Container     │    │ • Health Checks │
│ • SSL/TLS       │    │ • Auto-scaling  │    │ • Metrics       │
│ • Rate Limiting │    │ • Load Balanced │    │ • Alerting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Container Architecture

```dockerfile
# Multi-stage build for optimization
FROM python:3.12-slim as builder
# Build dependencies and application

FROM python:3.12-slim as runtime
# Runtime environment and application
```

### Environment Configuration

- **Development**: Auto-reload, debug logging
- **Staging**: Production-like with debug capabilities
- **Production**: Optimized for performance and security

## Scalability Considerations

### Horizontal Scaling

1. **Stateless Design**: No server-side state storage
2. **Load Balancing**: Multiple server instances
3. **Connection Pooling**: Efficient resource utilization
4. **Health Checks**: Automatic failover support

### Vertical Scaling

1. **Async Processing**: Efficient resource utilization
2. **Memory Management**: Optimized Python runtime
3. **Connection Limits**: Configurable connection pools
4. **Request Queuing**: Backpressure handling

## Integration Points

### External Dependencies

1. **Claude AI API**: Primary model access
2. **Claude CLI**: Authentication and configuration
3. **File System**: Configuration and logging
4. **Environment**: Runtime configuration

### Internal Integration

1. **FastAPI Framework**: HTTP server and routing
2. **Pydantic**: Data validation and serialization
3. **Claude SDK**: Official Python SDK integration
4. **Uvicorn**: ASGI server implementation
