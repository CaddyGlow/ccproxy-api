# Architecture Documentation

## System Overview

The Claude Code Proxy API Server is a personal FastAPI application that provides both Anthropic and OpenAI-compatible interfaces for Claude AI models. The system acts as a local translation layer between standard AI API formats and your Claude subscription, enabling seamless integration with your personal tools and applications while maintaining privacy and security.

## Core Architecture

### Personal Use Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Your Local     │    │  Docker         │    │  Claude Proxy   │
│  Applications   │────│  Container      │────│  Server         │
│                 │    │  (Isolation)    │    │                 │
│ • Scripts       │    │ • Security      │    │ • FastAPI App   │
│ • Notebooks     │    │ • Resource      │    │ • API Routes    │
│ • Web Apps      │    │   Limits        │    │ • Local Auth    │
│ • AI Tools      │    │ • Process       │    │ • Privacy       │
└─────────────────┘    │   Isolation     │    └─────────────────┘
         │              └─────────────────┘              │
         │              Localhost (127.0.0.1:8000)      │
         └─────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Your Claude CLI │
                                              │                 │
                                              │ • OAuth2 Auth   │
                                              │ • Local Config  │
                                              │ • Your Account  │
                                              └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Claude AI API   │
                                              │                 │
                                              │ • Your Subscription │
                                              │ • Model Access  │
                                              │ • Usage Tracking│
                                              └─────────────────┘
```

### Component Responsibilities for Personal Use

#### 1. API Layer (Your Interface)
- **Location**: `claude_code_proxy/api/`
- **Purpose**: Provides familiar API interfaces for your applications
- **Components**:
  - `v1/chat.py`: Anthropic-compatible endpoints for Claude tools
  - `openai/chat.py`: OpenAI-compatible endpoints for existing tools
  - `openai/models.py`: Model listing for compatibility

#### 2. Service Layer (Local Processing)
- **Location**: `claude_code_proxy/services/`
- **Purpose**: Handles local authentication and request processing
- **Components**:
  - `claude_client.py`: Integration with your Claude CLI
  - `streaming.py`: Real-time response streaming (Anthropic format)
  - `openai_streaming.py`: OpenAI format streaming for existing tools
  - `translator.py`: Seamless format conversion

#### 3. Data Models (Privacy & Validation)
- **Location**: `claude_code_proxy/models/`
- **Purpose**: Secure data validation and local processing
- **Components**:
  - `requests.py`: Request validation (Anthropic format)
  - `responses.py`: Response formatting (Anthropic format)
  - `openai_models.py`: OpenAI format compatibility
  - `errors.py`: Local error handling

#### 4. Configuration Management (Personal Settings)
- **Location**: `claude_code_proxy/config/`
- **Purpose**: Your personal configuration and preferences
- **Components**:
  - `settings.py`: Local configuration management
  - Personal Docker settings
  - Your Claude CLI path detection

#### 5. Security & Privacy
- **Location**: `claude_code_proxy/exceptions.py`
- **Purpose**: Secure error handling without data leakage
- **Features**:
  - Safe error responses
  - Local logging only
  - No external error reporting

## Data Flow Architecture

### Personal Request Processing Flow

```
Your Application Request (localhost)
     │
     ▼
┌─────────────────┐
│ Docker Container│
│  (Isolated)     │
│ • Local Access  │
│ • Privacy       │
│ • Security      │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  Local FastAPI  │
│                 │
│ • Request Logs  │
│ • Local CORS    │
│ • No Rate Limit│
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Request Models  │
│                 │
│ • Local Valid   │
│ • Type Safety   │
│ • Privacy Check │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Format Translator│
│                 │
│ • OpenAI → Claude│
│ • Your Tools ↔ API│
│ • Local Process │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Your Claude CLI │
│                 │
│ • Your Auth     │
│ • Your Account  │
│ • Local Config  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Response Format │
│                 │
│ • Your Format   │
│ • Local Stream  │
│ • Privacy Safe  │
└─────────────────┘
     │
     ▼
Response to Your Application
```

### Personal Streaming Response Flow

```
Your App Request (stream=true, localhost)
     │
     ▼
┌─────────────────┐
│ Local Container │
│                 │
│ • Secure Stream │
│ • Local Only    │
│ • Your Privacy  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Your Claude CLI │
│                 │
│ • Stream Mode   │
│ • Real-time     │
│ • Your Account  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Local Formatter │
│                 │
│ • Your Format   │
│ • Local Buffer  │
│ • Privacy Safe  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Stream to Your  │
│ Application     │
│ • Real-time     │
│ • Local Network │
│ • No External   │
└─────────────────┘
```

## Personal API Layer Architecture

### Local Endpoint Structure

```
localhost:8000/
├── /health                     # Your proxy health status
├── /v1/                       # For Claude-native tools
│   ├── chat/completions       # Direct Claude format
│   └── models                 # Available models from your account
└── /openai/v1/                # For existing OpenAI-compatible tools
    ├── chat/completions       # OpenAI format → Your Claude account
    └── models                 # Your models in OpenAI format
```

### Personal Request/Response Handling

#### Anthropic Format (v1/) - For Claude Tools
- **Request**: Native Claude API format
- **Response**: Direct Claude responses
- **Streaming**: Real-time streaming from your Claude subscription
- **Benefits**: Full feature access, direct integration

#### OpenAI Format (openai/v1/) - For Existing Tools
- **Request**: OpenAI API format (your existing tools work unchanged)
- **Response**: OpenAI-compatible format
- **Streaming**: OpenAI-style streaming
- **Benefits**: Drop-in replacement for OpenAI calls

## Personal Service Layer Design

### Personal Claude Client Service

```python
class ClaudeClient:
    """Personal service for your Claude account integration"""
    
    # Your local authentication
    # Private request processing
    # Local error handling
    # Personal streaming support
    # Your model access
```

**Personal Use Features**:
- Uses your existing Claude CLI authentication
- All processing happens locally on your machine
- No external logging or data sharing
- Direct access to your Claude subscription
- Personal usage tracking (stays local)

### Personal Streaming Services

#### Local Anthropic Streaming (`streaming.py`)
- Handles streaming from your Claude subscription
- Formats responses for your applications
- Maintains secure local connections
- Implements local error recovery

#### Personal OpenAI Streaming (`openai_streaming.py`)
- Converts your Claude responses to OpenAI format
- Works with your existing OpenAI-compatible tools
- Maintains tool compatibility without changing your workflow
- Handles local text chunking and formatting

### Personal Translation Service

The translator enables your existing tools to work with Claude:

```python
class OpenAITranslator:
    """Translates between your tools and Claude formats"""
    
    # Your OpenAI tools → Claude requests
    # Claude responses → Your tool format
    # Local parameter mapping
    # Private format conversion
```

## Personal Configuration Architecture

### Personal Settings Management

```python
class Settings(BaseSettings):
    """Your personal configuration management"""
    
    # Your local server settings
    # Your Claude CLI integration
    # Your Docker preferences
    # Your authentication setup
    # Your logging preferences
```

### Personal Configuration Sources (Priority Order)

1. **Your Environment Variables**: Your personal runtime settings
2. **Your Configuration Files**: Your `.env` and config files
3. **Safe Defaults**: Secure defaults for personal use
4. **Auto-detection**: Finds your Claude CLI automatically

### Personal Docker Integration

```python
class DockerSettings:
    """Your personal Docker configuration"""
    
    # Your container preferences
    # Your local volume mounts
    # Your environment setup
    # Your network security (localhost binding)
```

## Personal Error Handling Architecture

### Local Exception Hierarchy

```
ClaudeProxyError (Base - Your Local Errors)
├── ValidationError          # Your input validation
├── AuthenticationError      # Your Claude CLI auth issues  
├── PermissionError         # Local permission issues
├── NotFoundError           # Missing models/endpoints
├── ModelNotFoundError      # Model not in your subscription
├── TimeoutError            # Local timeout issues
└── ServiceUnavailableError # Claude service issues
```

### Private Error Response Format

#### Anthropic Format (No Data Leakage)
```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Safe error description (no personal data)"
  }
}
```

#### OpenAI Format (Privacy Safe)
```json
{
  "error": {
    "message": "Safe error description",
    "type": "invalid_request_error", 
    "param": "parameter_name",
    "code": "error_code"
  }
}
```

## Personal Security Architecture

### Local Authentication Flow

```
Your Application Request (localhost)
     │
     ▼
┌─────────────────┐
│ Local Container │
│                 │
│ • Localhost Only│
│ • No External   │
│ • Input Valid   │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Your Claude Auth│
│                 │
│ • Your CLI Auth │
│ • Your Account  │
│ • Local Config  │
└─────────────────┘
     │
     ▼
┌─────────────────┐
│ Safe Processing │
│                 │
│ • Local Only    │
│ • Your Privacy  │
│ • No Tracking   │
└─────────────────┘
```

### Personal Security Features

1. **Local Authentication**: Uses your existing Claude CLI setup
2. **Network Isolation**: Binds to localhost only (127.0.0.1)
3. **Container Isolation**: Docker provides process isolation
4. **Privacy Protection**: No external logging or data sharing
5. **Input Validation**: Local validation without data transmission

## Personal Performance Architecture

### Local Optimization Strategies

1. **Async Processing**: Efficient handling of your requests
2. **Local Connection Reuse**: Optimized connections to Claude
3. **Real-time Streaming**: Direct streaming from your subscription
4. **Memory Efficiency**: Optimized for personal computer resources
5. **Lightweight Setup**: Minimal resource usage for personal use

### Personal Monitoring Points

1. **Your Request Metrics**: Personal usage tracking (local only)
2. **Local Error Rates**: Monitor your setup health
3. **Stream Health**: Your streaming connection stability
4. **Container Resources**: Monitor Docker resource usage
5. **Claude CLI Health**: Your authentication and connection status

## Personal Deployment Architecture

### Local Personal Deployment

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Your Computer  │    │ Docker Container│    │ Your Local Apps │
│                 │    │                 │    │                 │
│ • Docker Engine │    │ • Claude Proxy  │    │ • Scripts       │
│ • Your Files    │    │ • Isolation     │    │ • Notebooks     │
│ • Claude CLI    │    │ • Local Network │    │ • Tools         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Personal Container Architecture

```dockerfile
# Lightweight build for personal use
FROM python:3.11-slim as builder
# Your dependencies and application

FROM python:3.11-slim as runtime
# Your runtime environment (minimal and secure)
```

### Personal Environment Configuration

- **Development**: Auto-reload enabled, verbose logging for learning
- **Daily Use**: Stable configuration, minimal logging
- **Isolated**: Docker container provides security and isolation

## Personal Use Considerations

### Personal Resource Management

1. **Single User Design**: Optimized for individual use
2. **Local Resource Limits**: Respects your computer's capabilities
3. **Memory Efficiency**: Minimal memory footprint for personal machines
4. **CPU Optimization**: Uses appropriate resources for your hardware

### Personal Scaling

1. **Efficient Processing**: Fast async handling for your requests
2. **Memory Management**: Optimized for personal computer resources
3. **Connection Management**: Maintains stable connection to Claude
4. **Request Handling**: Handles multiple personal requests efficiently

## Personal Integration Points

### Your Dependencies

1. **Claude AI API**: Access through your subscription
2. **Your Claude CLI**: Your local authentication and configuration
3. **Your File System**: Your local configuration and logs
4. **Your Environment**: Your personal runtime settings

### Your Local Integration

1. **FastAPI Framework**: Lightweight HTTP server for local use
2. **Pydantic**: Data validation for your requests
3. **Claude SDK**: Official integration with your Claude account
4. **Uvicorn**: Local ASGI server for your applications

## Summary

This architecture provides a secure, private, and efficient way to use Claude AI models locally on your personal computer. The Docker-based isolation ensures security while maintaining easy access to your Claude subscription through familiar API interfaces.

### Key Benefits

- **Privacy**: All processing happens locally on your machine
- **Security**: Docker isolation and localhost-only binding
- **Compatibility**: Works with both Anthropic and OpenAI API formats
- **Convenience**: Uses your existing Claude subscription
- **Flexibility**: Supports both streaming and non-streaming responses
- **Isolation**: Container-based security for Claude Code execution
