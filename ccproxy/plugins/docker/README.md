# Docker Plugin

Provides Docker-backed execution for CCProxy via CLI extensions.

## Highlights
- Wraps requests with `DockerAdapter` to run providers inside containers
- Extends the `ccproxy serve` CLI with Docker-specific arguments
- Applies CLI overrides to runtime configuration before adapter startup

## Configuration
- `DockerConfig` controls image, workspace, env vars, and volume mounts
- CLI flags override the configuration and are declared via `cli_arguments`
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin docker --config-class DockerConfig`

```toml
[plugins.docker]
# enabled = true
# docker_image = "anthropics/claude-cli:latest"
# docker_home_directory = "/home/user"
# docker_workspace_directory = "/workspace"
# docker_volumes = []
# docker_environment = []
# user_mapping_enabled = true
# user_uid = 1000
# user_gid = 1000
```

## Related Components
- `adapter.py`: executor that launches Docker containers
- `plugin.py`: runtime handling CLI context and overrides
- `config.py`: settings model for Docker execution
