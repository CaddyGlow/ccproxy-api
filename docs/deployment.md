# Deployment Guide

## Overview

This guide covers production deployment of the Claude Code Proxy API Server using various deployment strategies including Docker, container orchestration, and traditional server deployment.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **Python**: 3.11 or higher
- **Memory**: Minimum 512MB RAM, recommended 2GB+
- **CPU**: Minimum 1 vCPU, recommended 2+ vCPUs
- **Disk**: Minimum 1GB free space
- **Network**: Internet access for Claude API calls

### Required Dependencies

- Docker (for containerized deployment)
- Claude CLI (for authentication)
- Reverse proxy (nginx, Apache, or cloud load balancer)
- SSL certificate (for HTTPS)

## Docker Deployment

### Using Pre-built Image

#### Basic Deployment

```bash
# Pull the latest image
docker pull claude-code-proxy:latest

# Run the container
docker run -d \
  --name claude-proxy \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  -v ~/.config/claude:/root/.config/claude:ro \
  claude-code-proxy:latest
```

#### Production Deployment with Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  claude-proxy:
    image: claude-code-proxy:latest
    container_name: claude-proxy
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
      - WORKERS=4
    volumes:
      - ~/.config/claude:/root/.config/claude:ro
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    container_name: claude-proxy-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - claude-proxy
```

### Building Custom Image

#### Dockerfile

```dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && \
    uv sync --no-dev

FROM python:3.11-slim as runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash claude

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY claude_code_proxy/ ./claude_code_proxy/
COPY entrypoint.sh ./

# Set permissions
RUN chmod +x entrypoint.sh && \
    chown -R claude:claude /app

# Switch to non-root user
USER claude

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
ENTRYPOINT ["./entrypoint.sh"]
```

#### Build Script

```bash
#!/bin/bash
# build.sh

set -e

# Build the image
docker build \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg VCS_REF=$(git rev-parse --short HEAD) \
  --build-arg VERSION=$(git describe --tags --always) \
  -t claude-code-proxy:latest \
  -t claude-code-proxy:$(git describe --tags --always) \
  .

echo "Build completed successfully"
```

### Container Configuration

#### Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=4

# Claude Configuration
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Security
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Performance
RELOAD=false
```

#### Volume Mounts

```bash
# Claude CLI configuration (read-only)
-v ~/.config/claude:/root/.config/claude:ro

# Application logs
-v ./logs:/app/logs

# Custom configuration
-v ./config.json:/app/config.json:ro

# SSL certificates (if needed)
-v ./ssl:/app/ssl:ro
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: claude-proxy

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claude-proxy-config
  namespace: claude-proxy
data:
  HOST: "0.0.0.0"
  PORT: "8000"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-proxy
  namespace: claude-proxy
  labels:
    app: claude-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: claude-proxy
  template:
    metadata:
      labels:
        app: claude-proxy
    spec:
      containers:
      - name: claude-proxy
        image: claude-code-proxy:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: claude-proxy-config
        volumeMounts:
        - name: claude-config
          mountPath: /root/.config/claude
          readOnly: true
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: claude-config
        secret:
          secretName: claude-config-secret
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: claude-proxy-service
  namespace: claude-proxy
spec:
  selector:
    app: claude-proxy
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: claude-proxy-ingress
  namespace: claude-proxy
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: claude-proxy-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: claude-proxy-service
            port:
              number: 80
```

## Traditional Server Deployment

### System Service Setup

#### Create Service User

```bash
# Create dedicated user
sudo useradd --system --shell /bin/bash --home /opt/claude-proxy claude-proxy

# Create application directory
sudo mkdir -p /opt/claude-proxy
sudo chown claude-proxy:claude-proxy /opt/claude-proxy
```

#### Install Application

```bash
# Switch to service user
sudo -u claude-proxy -s

# Clone repository
cd /opt/claude-proxy
git clone https://github.com/your-org/claude-proxy.git .

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -e .
```

#### Systemd Service

Create `/etc/systemd/system/claude-proxy.service`:

```ini
[Unit]
Description=Claude Code Proxy API Server
After=network.target
Wants=network-online.target

[Service]
Type=exec
User=claude-proxy
Group=claude-proxy
WorkingDirectory=/opt/claude-proxy
Environment=PATH=/opt/claude-proxy/venv/bin
Environment=HOST=127.0.0.1
Environment=PORT=8000
Environment=LOG_LEVEL=INFO
Environment=WORKERS=4
ExecStart=/opt/claude-proxy/venv/bin/python -m uvicorn claude_code_proxy.main:app --host 127.0.0.1 --port 8000 --workers 4
ExecReload=/bin/kill -s HUP $MAINPID
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Service Management

```bash
# Enable and start service
sudo systemctl enable claude-proxy
sudo systemctl start claude-proxy

# Check status
sudo systemctl status claude-proxy

# View logs
sudo journalctl -u claude-proxy -f
```

## Load Balancer Configuration

### Nginx Configuration

Create `/etc/nginx/sites-available/claude-proxy`:

```nginx
upstream claude_proxy {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=claude_api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=claude_burst:10m rate=100r/m;

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/claude-proxy.crt;
    ssl_certificate_key /etc/ssl/private/claude-proxy.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate Limiting
    limit_req zone=claude_api burst=20 nodelay;
    limit_req zone=claude_burst burst=50 nodelay;

    # Proxy Configuration
    location / {
        proxy_pass http://claude_proxy;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Streaming support
        proxy_buffering off;
        proxy_cache off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Health check
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://claude_proxy;
        proxy_set_header Host $host;
    }

    # Logging
    access_log /var/log/nginx/claude-proxy.access.log;
    error_log /var/log/nginx/claude-proxy.error.log;
}
```

### HAProxy Configuration

```haproxy
global
    daemon
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    option dontlognull

frontend claude_proxy_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/claude-proxy.pem
    redirect scheme https if !{ ssl_fc }
    
    # Rate limiting
    stick-table type ip size 100k expire 30s store http_req_rate(10s)
    http-request track-sc0 src
    http-request reject if { sc_http_req_rate(0) gt 20 }
    
    default_backend claude_proxy_backend

backend claude_proxy_backend
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server proxy1 127.0.0.1:8000 check
    server proxy2 127.0.0.1:8001 check
    server proxy3 127.0.0.1:8002 check
```

## Environment Configuration

### Production Environment Variables

```bash
# .env.production
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=4
RELOAD=false

# Security
CORS_ORIGINS=https://yourdomain.com

# Performance
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Monitoring
HEALTH_CHECK_INTERVAL=30
```

### Configuration File

Create `config.production.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": false
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "/var/log/claude-proxy/app.log"
  },
  "security": {
    "cors_origins": ["https://yourdomain.com"],
    "rate_limit": {
      "requests_per_minute": 100,
      "burst_size": 20
    }
  },
  "claude": {
    "cli_path": "/usr/local/bin/claude",
    "timeout": 300
  }
}
```

## Health Monitoring

### Health Check Endpoint

The `/health` endpoint provides service status:

```json
{
  "status": "healthy",
  "service": "claude-proxy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### Monitoring Script

```bash
#!/bin/bash
# health-check.sh

ENDPOINT="http://localhost:8000/health"
TIMEOUT=10

response=$(curl -s -w "%{http_code}" -o /dev/null --max-time $TIMEOUT "$ENDPOINT")

if [ "$response" = "200" ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP $response)"
    exit 1
fi
```

### Prometheus Metrics

Add monitoring endpoints (if metrics are enabled):

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'claude-proxy'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

## Security Considerations

### SSL/TLS Configuration

1. **Use strong SSL certificates** (Let's Encrypt or commercial)
2. **Enable HSTS** headers
3. **Disable weak protocols** (TLS 1.0, 1.1)
4. **Use strong cipher suites**

### Network Security

1. **Firewall rules** to restrict access
2. **Rate limiting** to prevent abuse
3. **IP whitelisting** for admin endpoints
4. **VPN access** for internal services

### Application Security

1. **Regular updates** of dependencies
2. **Security scanning** of container images
3. **Log monitoring** for suspicious activity
4. **Access control** for configuration files

## Backup and Recovery

### Configuration Backup

```bash
#!/bin/bash
# backup-config.sh

backup_dir="/backup/claude-proxy/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# Backup configuration files
cp /opt/claude-proxy/.env "$backup_dir/"
cp /opt/claude-proxy/config.json "$backup_dir/"
cp -r ~/.config/claude "$backup_dir/claude-config"

# Backup logs
cp -r /var/log/claude-proxy "$backup_dir/logs"

echo "Backup completed: $backup_dir"
```

### Disaster Recovery

1. **Automated backups** of configuration
2. **Infrastructure as Code** for reproducible deployments
3. **Multi-region deployment** for high availability
4. **Recovery procedures** documentation

## Performance Optimization

### Application Tuning

```bash
# Performance environment variables
WORKERS=4  # Number of CPU cores
WORKER_CLASS=uvicorn.workers.UvicornWorker
WORKER_CONNECTIONS=1000
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50
```

### System Tuning

```bash
# /etc/sysctl.conf
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 1024
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_fin_timeout = 30
```

### Container Resource Limits

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check for services using port 8000
2. **Claude CLI authentication**: Verify Claude CLI setup
3. **Memory issues**: Monitor container memory usage
4. **Network connectivity**: Test Claude API access

### Log Analysis

```bash
# View application logs
docker logs claude-proxy

# Follow logs in real-time
docker logs -f claude-proxy

# Search for errors
docker logs claude-proxy 2>&1 | grep ERROR

# System service logs
journalctl -u claude-proxy -f
```

### Performance Monitoring

```bash
# Container resource usage
docker stats claude-proxy

# System resource usage
htop
iotop
netstat -tulpn
```