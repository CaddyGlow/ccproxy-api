# Stage 1: Claude install
# FROM node:18-slim AS node-deps
#
# WORKDIR /app
#
# # Install pnpm globally
# RUN npm install -g pnpm
#
# # Copy package.json and install JavaScript dependencies
# COPY package.json ./
# RUN pnpm install

# Stage 1: Install bun from the official image
FROM oven/bun:1-slim AS bun-deps
RUN bun install -g @anthropic-ai/claude-code

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
  curl wget ripgrep fd-find exa sed mawk procps\
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

# Copy Node.js binaries from node-deps stage
# COPY --from=node-deps /usr/local/bin/node /usr/local/bin/node
# COPY --from=node-deps /usr/local/bin/npm /usr/local/bin/npm
# COPY --from=node-deps /usr/local/bin/npx /usr/local/bin/npx
# COPY --from=node-deps /usr/local/bin/pnpm /usr/local/bin/pnpm
# COPY --from=node-deps /usr/local/lib/node_modules /usr/local/lib/node_modules
# We have to copy the entire /usr/local that seem to be
# more realiable
#COPY --from=node-deps /usr/local /usr/local

# Copy bun binaries from bun-deps stage and link to node
COPY --from=bun-deps /usr/local/bin/bun /usr/local/bin/
COPY --from=bun-deps /usr/local/bin/bunx /usr/local/bin/
RUN ln -s /usr/local/bin/bun /usr/local/bin/node && ln -s /usr/local/bin/bunx /usr/local/bin/npx

# Install package for claude and link to claude bin
COPY --from=bun-deps /root/.bun/install/global /app/bun_global
RUN ln -s /app/bun_global/node_modules/\@anthropic-ai/claude-code/cli.js /usr/local/bin/claude

COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Copy Python application from builder
COPY --from=builder /app /app

WORKDIR /app

# ENV PATH="/app/.venv/bin:/app/node_modules/.bin:$PATH"
ENV PATH="/app/.venv/bin:/app/bun_global/bin:$PATH"
ENV PYTHONPATH=/app
ENV SERVER__HOST=0.0.0.0
ENV SERVER__PORT=8000

EXPOSE ${SERVER__PORT:-8000}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${SERVER__PORT:-8000}/health || exit 1

# Entrypoint used to create user and set
# user home folder
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["ccproxy"]
