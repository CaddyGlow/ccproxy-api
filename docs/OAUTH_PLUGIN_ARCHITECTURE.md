# OAuth Plugin Architecture

## Overview

The OAuth system in CCProxy follows a plugin-owned, registry-based pattern that allows each plugin to manage its own OAuth implementation while benefiting from shared infrastructure.

## Architecture Components

### 1. Core Infrastructure (`ccproxy/auth/oauth/`)

- **`registry.py`** - Central registry for dynamic provider discovery
- **`protocol.py`** - OAuth protocol definitions that plugins implement  
- **`router.py`** - Unified OAuth endpoints that delegate to plugins
- **`session.py`** - Session management for OAuth flows
- **`base.py`** - Generic base OAuth client with PKCE support

### 2. Plugin Implementation (e.g., `plugins/claude_api/oauth/`)

- **`provider.py`** - OAuth provider implementing the protocol
- **`client.py`** - Provider-specific OAuth client
- **`config.py`** - OAuth configuration
- **`storage.py`** - Token storage implementation

## Implementation Guide

### Step 1: Create OAuth Module Structure

```
plugins/your_plugin/oauth/
├── __init__.py
├── config.py      # OAuth configuration
├── client.py      # OAuth client implementation
├── provider.py    # Provider for registry
└── storage.py     # Token storage
```

### Step 2: Define OAuth Configuration

```python
# config.py
from pydantic import BaseModel, Field

class YourOAuthConfig(BaseModel):
    client_id: str = Field(default="your_client_id")
    redirect_uri: str = Field(default="http://localhost:9999/oauth/your-plugin/callback")
    base_url: str = Field(default="https://api.example.com")
    authorize_url: str = Field(default="https://api.example.com/oauth/authorize")
    token_url: str = Field(default="https://api.example.com/oauth/token")
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])
    use_pkce: bool = Field(default=True)
```

### Step 3: Implement OAuth Client

```python
# client.py
from typing import Any
from ccproxy.auth.oauth.base import BaseOAuthClient
from ccproxy.auth.models import YourCredentials
from .config import YourOAuthConfig

class YourOAuthClient(BaseOAuthClient[YourCredentials]):
    def __init__(self, config: YourOAuthConfig, storage=None):
        self.oauth_config = config
        super().__init__(
            client_id=config.client_id,
            redirect_uri=config.redirect_uri,
            base_url=config.base_url,
            scopes=config.scopes,
            storage=storage,
        )

    def _get_auth_endpoint(self) -> str:
        return self.oauth_config.authorize_url

    def _get_token_endpoint(self) -> str:
        return self.oauth_config.token_url

    async def parse_token_response(self, data: dict[str, Any]) -> YourCredentials:
        # Parse provider-specific token response
        return YourCredentials(...)
```

### Step 4: Create OAuth Provider

```python
# provider.py
from typing import Any
from ccproxy.auth.oauth.registry import OAuthProviderInfo
from .client import YourOAuthClient
from .config import YourOAuthConfig
from .storage import YourTokenStorage

class YourOAuthProvider:
    def __init__(self, config=None, storage=None):
        self.config = config or YourOAuthConfig()
        self.storage = storage or YourTokenStorage()
        self.client = YourOAuthClient(self.config, self.storage)

    @property
    def provider_name(self) -> str:
        return "your-plugin"

    @property
    def provider_display_name(self) -> str:
        return "Your Service"

    @property
    def supports_pkce(self) -> bool:
        return self.config.use_pkce

    async def get_authorization_url(
        self, state: str, code_verifier: str | None = None
    ) -> str:
        # Generate authorization URL
        auth_url, _, _ = await self.client.authenticate(code_verifier, state)
        return auth_url

    async def handle_callback(
        self, code: str, state: str, code_verifier: str | None = None
    ) -> Any:
        # Handle OAuth callback
        credentials = await self.client.handle_callback(code, state, code_verifier or "")
        if self.storage:
            await self.storage.save_credentials(credentials)
        return credentials

    async def refresh_access_token(self, refresh_token: str) -> Any:
        # Refresh tokens
        credentials = await self.client.refresh_token(refresh_token)
        if self.storage:
            await self.storage.save_credentials(credentials)
        return credentials

    def get_provider_info(self) -> OAuthProviderInfo:
        return OAuthProviderInfo(
            name=self.provider_name,
            display_name=self.provider_display_name,
            description="OAuth for Your Service",
            supports_pkce=self.supports_pkce,
            scopes=self.config.scopes,
            is_available=True,
            plugin_name="your_plugin",
        )
```

### Step 5: Implement Token Storage

```python
# storage.py
from pathlib import Path
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.auth.models import YourCredentials

class YourTokenStorage(TokenStorage[YourCredentials]):
    def __init__(self, storage_path: Path | None = None):
        if storage_path is None:
            storage_path = Path.home() / ".ccproxy" / "your_credentials.json"
        self.file_path = storage_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    async def save(self, credentials: YourCredentials) -> bool:
        # Save credentials to file
        data = credentials.model_dump(mode="json", exclude_none=True)
        self.file_path.write_text(json.dumps(data, indent=2))
        return True

    async def load(self) -> YourCredentials | None:
        # Load credentials from file
        if not self.file_path.exists():
            return None
        data = json.loads(self.file_path.read_text())
        return YourCredentials.model_validate(data)

    async def exists(self) -> bool:
        return self.file_path.exists()

    async def delete(self) -> bool:
        if self.file_path.exists():
            self.file_path.unlink()
            return True
        return False

    def get_location(self) -> str:
        return str(self.file_path)
```

### Step 6: Register Provider in Plugin

```python
# plugin.py
from ccproxy.plugins.declaration import PluginManifest, PluginFactory

def _create_oauth_provider() -> Any:
    from plugins.your_plugin.oauth import YourOAuthProvider
    return YourOAuthProvider()

manifest = PluginManifest(
    name="your_plugin",
    version="1.0.0",
    description="Your plugin with OAuth",
    oauth_provider_factory=_create_oauth_provider,
    # ... other manifest fields
)

factory = PluginFactory(manifest)
```

## OAuth Flow

### 1. User Initiates Login
```
GET /oauth/your-plugin/login
```

### 2. System Creates Session and Redirects
- Generates PKCE code verifier and state
- Stores session with provider and code verifier
- Gets authorization URL from provider
- Redirects user to provider's OAuth page

### 3. Provider Handles Callback
```
GET /oauth/your-plugin/callback?code=xxx&state=yyy
```

### 4. System Completes Flow
- Validates state parameter
- Retrieves session data
- Exchanges code for tokens using PKCE verifier
- Stores credentials
- Returns success page

## Token Management

### Refresh Tokens
```
POST /oauth/your-plugin/refresh
{
    "refresh_token": "..."
}
```

### Revoke Tokens
```
POST /oauth/your-plugin/revoke
{
    "token": "..."
}
```

## Security Features

### PKCE Support
The base OAuth client automatically handles PKCE:
- Generates secure code verifier
- Calculates SHA256 code challenge
- Validates during token exchange

### Session Management
- Secure state parameter generation
- Time-limited sessions (5 minutes default)
- Automatic session cleanup

### Token Storage
- Encrypted storage options available
- File permissions (600) for local storage
- Atomic file operations

## Benefits

1. **Encapsulation**: Each plugin owns its OAuth implementation
2. **Flexibility**: Support for different OAuth flows
3. **Type Safety**: Strong typing with credential models
4. **Reusability**: Shared base classes reduce boilerplate
5. **Discovery**: Dynamic provider registration
6. **Maintainability**: OAuth logic stays with its provider

## Testing

### Unit Tests
```python
async def test_oauth_flow():
    provider = YourOAuthProvider()

    # Test authorization URL generation
    auth_url = await provider.get_authorization_url("test_state", "test_verifier")
    assert "client_id=" in auth_url

    # Test callback handling
    credentials = await provider.handle_callback("test_code", "test_state", "test_verifier")
    assert credentials is not None
```

### Integration Tests
```python
async def test_oauth_endpoints(client: AsyncClient):
    # Test provider discovery
    response = await client.get("/oauth/providers")
    assert "your-plugin" in response.json()

    # Test login initiation
    response = await client.get("/oauth/your-plugin/login")
    assert response.status_code == 302
```

## Migration Guide

### From Centralized OAuth

1. **Move OAuth code to plugin:**
   - Create `oauth/` directory in plugin
   - Move provider-specific code from `ccproxy/auth/oauth/providers/`

2. **Update imports:**
   - Change from `ccproxy.auth.oauth.providers.your_provider`
   - To `plugins.your_plugin.oauth`

3. **Register provider:**
   - Add `oauth_provider_factory` to manifest
   - Remove old provider registration

4. **Update configuration:**
   - Move OAuth config to plugin settings
   - Update environment variables if needed

5. **Test thoroughly:**
   - Verify OAuth flow works
   - Check token refresh
   - Test error handling

## Troubleshooting

### Provider Not Listed
- Check provider is registered in manifest
- Verify factory creates provider correctly
- Check for import errors in logs

### OAuth Flow Fails
- Verify redirect URI matches configuration
- Check PKCE settings match provider requirements
- Review session timeout settings
- Check for CORS issues

### Token Storage Issues
- Verify file permissions
- Check storage path exists
- Review storage implementation
- Check for serialization errors
