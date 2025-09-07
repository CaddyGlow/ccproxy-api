# Debugging with HTTP Proxy

This guide explains how to use HTTP proxies for debugging requests made by the CCProxy API server.

## Overview

The CCProxy API server supports standard HTTP proxy environment variables, allowing you to intercept and debug HTTP/HTTPS traffic using tools like:

- [mitmproxy](https://mitmproxy.org/)
- [Fiddler](https://www.telerik.com/fiddler)
- Corporate proxies

## Setting Up Proxy

### Basic Proxy Configuration

Set the appropriate environment variables before starting the server:

```bash
# For HTTP and HTTPS traffic
export HTTP_PROXY=http://localhost:8888
export HTTPS_PROXY=http://localhost:8888

# Or use ALL_PROXY for both protocols
export ALL_PROXY=http://localhost:8888

# Start the server
make dev
```

### Using mitmproxy

1. Install mitmproxy:
   ```bash
   pip install mitmproxy
   ```

2. Start mitmproxy:
   ```bash
   mitmproxy
   # or for web interface
   mitmweb
   ```

3. Export the proxy settings:
   ```bash
   export HTTPS_PROXY=http://localhost:8080
   ```

4. Install mitmproxy's CA certificate (see below)

## SSL/TLS Certificate Configuration

When using HTTPS proxies that perform SSL interception, you'll need to configure the proxy's root CA certificate.

### Option 1: Using Custom CA Bundle (Recommended)

```bash
# For mitmproxy
export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem

# Or use SSL_CERT_FILE
export SSL_CERT_FILE=/path/to/your/ca-bundle.pem

# Start the server
make dev
```

### Option 2: Disable SSL Verification (Development Only)

**WARNING**: This is insecure and should only be used for local development.

```bash
export SSL_VERIFY=false
make dev
```

### Installing Proxy CA Certificates

#### mitmproxy
```bash
# The CA certificate is typically located at:
~/.mitmproxy/mitmproxy-ca-cert.pem

# Set the environment variable
export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem
```

#### Charles Proxy
1. In Charles: Help > SSL Proxying > Save Charles Root Certificate
2. Save as PEM format
3. Set: `export REQUESTS_CA_BUNDLE=/path/to/charles-ca-cert.pem`

## Debugging Example

Here's a complete example using mitmproxy:

```bash
# Terminal 1: Start mitmproxy
mitmweb --listen-port 8888

# Terminal 2: Configure and run the server
export HTTPS_PROXY=http://localhost:8888
export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem
make dev

# Terminal 3: Make a test request
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-opus-20240229", "messages": [{"role": "user", "content": "Hello"}]}'
```

You should now see the request to `api.anthropic.com` in the mitmproxy web interface.

## Testing Proxy Configuration

Use the provided debug script to test your proxy setup:

```bash
# Set proxy environment variables
export HTTPS_PROXY=http://localhost:8888
export REQUESTS_CA_BUNDLE=~/.mitmproxy/mitmproxy-ca-cert.pem

# Run the debug script
python examples/proxy_debug.py
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `HTTP_PROXY` | Proxy for HTTP requests | `http://proxy.company.com:8080` |
| `HTTPS_PROXY` | Proxy for HTTPS requests | `http://proxy.company.com:8080` |
| `ALL_PROXY` | Proxy for all protocols | `http://proxy.company.com:8080` |
| `REQUESTS_CA_BUNDLE` | Path to CA certificate bundle | `/path/to/ca-bundle.pem` |
| `SSL_CERT_FILE` | Alternative to REQUESTS_CA_BUNDLE | `/path/to/ca-bundle.pem` |
| `SSL_VERIFY` | Enable/disable SSL verification | `true` or `false` |

## Troubleshooting

### SSL Certificate Errors

If you see SSL certificate verification errors:

1. Ensure the proxy's CA certificate is properly installed
2. Verify the certificate path is correct
3. Check that the certificate file is readable

### Proxy Connection Errors

1. Verify the proxy is running and accessible
2. Check the proxy URL format (should be `http://` even for HTTPS proxying)
3. Ensure no firewall is blocking the connection

### No Traffic in Proxy

1. Verify environment variables are set correctly
2. Restart the server after setting environment variables
3. Check that the proxy is configured to intercept HTTPS traffic

## OpenAI Format Endpoints

When using tools like Aider that expect OpenAI-formatted responses, ensure you're using the correct endpoint:

### Endpoint Differences

- **`/openai/v1/chat/completions`** - Reverse proxy endpoint that returns **Anthropic format**
- **`/cc/openai/v1/chat/completions`** - Claude Code SDK endpoint that returns **OpenAI format**

### Configuring Aider

To use Aider with the Claude Code Proxy, configure it to use the correct endpoint:

```bash
# Correct - Uses OpenAI format transformation
export OPENAI_API_BASE=http://localhost:8000/cc/openai/v1
aider

# Incorrect - Returns raw Anthropic format
# export OPENAI_API_BASE=http://localhost:8000/openai/v1
```

### Testing Endpoint Format

You can verify the endpoint format using the test script:

```bash
# This will show the difference between endpoints
python test_endpoint_difference.py
```

## Security Considerations

- **Never disable SSL verification in production**
- Only use proxy interception for debugging in development environments
- Be cautious with proxy credentials in environment variables
- Clear proxy settings when not debugging to avoid accidental traffic interception
