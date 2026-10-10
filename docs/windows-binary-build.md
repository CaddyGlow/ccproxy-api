# Windows Binary Build

## Overview

The Windows binary (`.zip` archive with `x86_64-pc-windows-msvc` triple naming) is **not built in this repository**. It is built by CaddyGlow's reusable CI pipeline in the separate repository:

- **Repository:** <https://github.com/CaddyGlow/homebrew-packages>
- **Workflow:** `.github/workflows/python-build.yml`
- **Action (setup):** `.github/actions/setup-python-env/action.yml`
- **Action (freeze):** `.github/actions/package-python-pyinstaller/action.yml`
- **Tool:** PyInstaller (`--onefile` mode)

## Build Flow

1. **Trigger:** `python-ci.yml` in `CaddyGlow/ccproxy-api` is triggered on tag pushes (`v*`) or `workflow_dispatch`
2. **Setup:** The `setup-python-env` action runs `uv sync --all-groups` to install the project and its dependency groups
3. **Freeze:** The `package-python-pyinstaller` action runs `uv pip install pyinstaller` followed by:
   ```bash
   pyinstaller --onefile --name "ccproxy" --noconfirm --clean \
     --collect-all ccproxy --hidden-import=ccproxy.cli.main \
     --collect-submodules ccproxy.plugins \
     __pyinstaller_entry__.py
   ```
4. **Package:** The resulting binary is packaged into a `.zip` or `.tar.gz` archive with version and target triple in the filename
5. **Release:** `python-release.yml` creates a GitHub Release from the workflow run artifacts

## The Bug: Missing Auth Plugins

### Symptom

```powershell
.\ccproxy.exe auth login claude
```

```
[ERROR] No auth providers installed in this CCProxy release
```

`.\ccproxy.exe auth providers` returns:

```
No OAuth providers found.
```

### Root Cause

The `pyproject.toml` defines:

1. **Entry points** under `[project.entry-points."ccproxy.plugins"]` — these register plugins like `oauth_claude`, `claude_api`, etc.
2. **An optional extra** `plugins-claude = ["claude-agent-sdk>=0.1.4", "qrcode>=8.2", "keyring>=25.7.0"]`

The build's `setup-python-env` action runs `uv sync --all-groups`, which installs the core package and its **dependency groups** (e.g., `dev`, `test`) but **not** the `[project.optional-dependencies]` extras. The `plugins-claude` dependencies are therefore missing from the frozen environment.

Additionally, in a PyInstaller one-file binary, the `.dist-info/entry_points.txt` metadata is not available at runtime, so `importlib.metadata.entry_points()` returns nothing. Even though `--collect-submodules ccproxy.plugins` bundles all plugin source files, the entry-point discovery mechanism finds zero plugins to load.

The plugin code uses `try/except ImportError` to handle missing optional dependencies, which means a plugin like `oauth_claude` imports successfully but can't perform its function (OAuth flows need `qrcode` for the device-code flow URL display, and `keyring` for secure token storage on Windows).

### Who Has the Build Script

The package-and-freeze logic lives entirely in `CaddyGlow/homebrew-packages`:

| File | Purpose |
|------|---------|
| `.github/workflows/python-build.yml` | Reusable workflow called by `python-ci.yml` |
| `.github/actions/setup-python-env/action.yml` | Sets up Python + uv + deps |
| `.github/actions/package-python-pyinstaller/action.yml` | Freezes binary with PyInstaller |
| `.github/actions/load-python-target-matrix/action.yml` | Provides target matrix (windows/linux/macos) |

This repository (`ClintonSarkar/ccproxy-api`) does **NOT** contain any Windows binary build scripts.

### Required Fix (in CaddyGlow/homebrew-packages)

The one-line fix is in `.github/actions/setup-python-env/action.yml`. Change the "Install dependencies" step from:

```yaml
    - name: Install dependencies
      shell: bash
      run: |
        if [ -f "pyproject.toml" ]; then
          uv sync --all-groups
        elif [ -f "requirements.txt" ]; then
          uv pip install -r requirements.txt
        else
          echo "::warning::No pyproject.toml or requirements.txt found"
        fi
```

to:

```yaml
    - name: Install dependencies (with extras for bundled plugins)
      shell: bash
      run: |
        if [ -f "pyproject.toml" ]; then
          uv sync --all-groups
          uv pip install ".[plugins-claude,plugins-codex]"
        elif [ -f "requirements.txt" ]; then
          uv pip install -r requirements.txt
        else
          echo "::warning::No pyproject.toml or requirements.txt found"
        fi
```

This installs both the core dependencies (`--all-groups`) and the optional extras that the bundled auth plugins need at runtime.

### Additional Safety: PyInstaller `.spec` File (Optional)

For a more robust fix, create a PyInstaller `.spec` file in the repo root that explicitly lists all hidden imports. The current CI passes dynamic flags via the shell script; a `.spec` file would make the freeze step more explicit and prevent future regressions.

### Workaround for Users

Until a fixed binary is released, install via Python package manager:

```powershell
# Install with pipx (recommended)
pipx install "ccproxy-api[plugins-claude,plugins-codex]"

# Or with uv
uv tool install "ccproxy-api[plugins-claude,plugins-codex]"

# Verify
ccproxy auth providers
```

## Verification

To check if a binary has working auth plugins:

```powershell
.\ccproxy.exe auth providers
```

A working build should show `oauth_claude` and `oauth_codex` in the list.

## Related Issue

- GitHub: <https://github.com/CaddyGlow/ccproxy-api/issues/75>

---

*Documented 2026-07-16 after investigating Windows binary build pipeline*
