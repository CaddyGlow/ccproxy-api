# Stage 1: Node.js dependencies
FROM node:18-slim AS node-deps

WORKDIR /app

# Install pnpm globally
RUN npm install -g pnpm

# Copy package.json and install JavaScript dependencies
COPY package.json ./
RUN pnpm install

# Stage 2: Python builder
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install git with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y \
  git \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --locked --no-install-project --no-dev

# Copy application code
COPY . /app

# Install the project
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --no-dev

# Stage 3: Runtime
FROM python:3.11-slim-bookworm

# Install system dependencies with apt cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Copy Node.js binaries from node-deps stage
COPY --from=node-deps /usr/local/bin/node /usr/local/bin/node
COPY --from=node-deps /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node-deps /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node-deps /usr/local/bin/pnpm /usr/local/bin/pnpm
COPY --from=node-deps /usr/local/lib/node_modules /usr/local/lib/node_modules

# Copy Python application from builder
COPY --from=builder /app /app

# Copy Node.js dependencies from first stage
COPY --from=node-deps /app/node_modules /app/node_modules
COPY --from=node-deps /app/package.json /app/package.json

# Set working directory
WORKDIR /app

# Set environment variables
ENV PATH="/app/.venv/bin:/app/node_modules/.bin:$PATH"
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py", "serve"]
