{
  description = "Development environment with Python, JavaScript, and Playwright";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # System libraries
        systemLibs = with pkgs; [
          # stdenv.cc.cc.lib
          # glibc
          # zlib
          # stdenv
        ];

        allLibs = systemLibs;
      in
      {
        packages.default = pkgs.python3Packages.buildPythonPackage rec {
          pname = "ccproxy-api";
          version = "0.2.0";
          pyproject = true;

          src = ../.;

          build-system = with pkgs.python3Packages; [
            hatchling
            hatch-vcs
          ];

          dependencies = with pkgs.python3Packages; [
            aiofiles
            fastapi
            httpx
            pydantic
            pydantic-settings
            rich
            rich-toolkit
            structlog
            typer
            typing-extensions
            uvicorn
            packaging
            pyjwt
            sortedcontainers
          ];

          optional-dependencies = with pkgs.python3Packages; {
            # Plugin dependencies grouped as in pyproject.toml
            plugins-claude = [
              # claude-code-sdk would need to be packaged separately
              qrcode
            ];
            plugins-codex = [
              qrcode
              pyjwt
            ];
            plugins-storage = [
              sqlmodel
              sqlalchemy
              # duckdb-engine and duckdb would need special handling
            ];
            test = [
              mypy
              pytest
              pytest-asyncio
              pytest-cov
              pytest-timeout
              # pytest-env
              # pytest-httpx
              # pytest-xdist
            ];
            dev = [
              # ruff would need to be available
              # pre-commit
              mypy
              # tox
              # bandit
            ];
          };

          # Skip tests during build since they require external services
          doCheck = false;

          pythonImportsCheck = [ "ccproxy" ];

          meta = with pkgs.lib; {
            description = "API server that provides an Anthropic and OpenAI compatible interface over Claude Code";
            homepage = "https://github.com/anthropics/ccproxy-api";
            license = licenses.mit; # Adjust based on actual license
            maintainers = [ ];
            platforms = platforms.unix;
          };
        };

        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/ccproxy-api";
        };

        devShells.default = pkgs.mkShell {
          buildInputs =
            with pkgs;
            [
              # Python with uv
              python3
              uv
              pyright

              # JavaScript tools
              nodejs
              bun
              pnpm

              # System tools
              bashInteractive
              duckdb
              chromium
              udev
              cacert
            ]
            ++ allLibs;

          shellHook = ''
            # Set up LD_LIBRARY_PATH for Playwright and other native dependencies
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath allLibs}:$LD_LIBRARY_PATH"
            # Optional: Set up uv if you want it to manage Python versions
            # export UV_PYTHON_PREFERENCE=system
            # Ensure HTTPS clients (e.g., mkdocs) can validate certificates
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

          '';

        };
      }
    );
}
