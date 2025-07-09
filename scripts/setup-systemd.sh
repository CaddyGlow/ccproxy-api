#!/usr/bin/env bash
set -euo pipefail

# Claude Code Proxy API - Systemd Service Setup Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="ccproxy"
USER_SERVICE_DIR="$HOME/.config/systemd/user"
TEMPLATE_FILE="$PROJECT_DIR/systemd/ccproxy.service.template"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect uv installation
detect_uv() {
    if command -v uv &> /dev/null; then
        command -v uv
    elif [ -f "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        echo "$HOME/.cargo/bin/uv"
    else
        return 1
    fi
}

# Function to setup systemd service
setup_service() {
    print_info "Setting up systemd user service for Claude Code Proxy API"

    # Check if template exists
    if [ ! -f "$TEMPLATE_FILE" ]; then
        print_error "Template file not found: $TEMPLATE_FILE"
        exit 1
    fi

    # Detect UV path
    UV_PATH=$(detect_uv) || {
        print_error "uv not found. Please install uv first: https://github.com/astral-sh/uv"
        exit 1
    }
    print_info "Found uv at: $UV_PATH"

    # Create systemd user directory if it doesn't exist
    mkdir -p "$USER_SERVICE_DIR"

    # Get user input for configuration
    print_info "Please provide the following configuration:"

    read -p "Working directory [$PROJECT_DIR]: " WORKING_DIR
    WORKING_DIR="${WORKING_DIR:-$PROJECT_DIR}"

    read -p "Main Python file [main.py]: " MAIN_PY
    MAIN_PY="${MAIN_PY:-main.py}"
    MAIN_PY_PATH="$WORKING_DIR/$MAIN_PY"

    # Check if main.py exists
    if [ ! -f "$MAIN_PY_PATH" ]; then
        print_warn "Main file not found at: $MAIN_PY_PATH"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Detect user PATH
    USER_PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"
    read -p "PATH environment [$USER_PATH]: " CUSTOM_PATH
    USER_PATH="${CUSTOM_PATH:-$USER_PATH}"

    # Additional environment variables
    print_info "Additional environment variables (optional)"
    print_info "Format: KEY=VALUE (press Enter to skip)"
    EXTRA_ENV=""
    while true; do
        read -p "Environment variable (or press Enter to finish): " ENV_VAR
        if [ -z "$ENV_VAR" ]; then
            break
        fi
        if [[ "$ENV_VAR" =~ ^[A-Za-z_][A-Za-z0-9_]*=.+$ ]]; then
            EXTRA_ENV="${EXTRA_ENV}Environment=\"${ENV_VAR}\"\n"
        else
            print_warn "Invalid format. Use KEY=VALUE"
        fi
    done

    # Service name
    read -p "Service name [$SERVICE_NAME]: " CUSTOM_SERVICE_NAME
    SERVICE_NAME="${CUSTOM_SERVICE_NAME:-$SERVICE_NAME}"

    # Generate service file
    SERVICE_FILE="$USER_SERVICE_DIR/${SERVICE_NAME}.service"

    print_info "Generating service file..."
    sed -e "s|{{WORKING_DIR}}|$WORKING_DIR|g" \
        -e "s|{{UV_PATH}}|$UV_PATH|g" \
        -e "s|{{MAIN_PY_PATH}}|$MAIN_PY_PATH|g" \
        -e "s|{{USER_PATH}}|$USER_PATH|g" \
        -e "s|{{USER_HOME}}|$HOME|g" \
        -e "s|{{EXTRA_ENV}}|$EXTRA_ENV|g" \
        "$TEMPLATE_FILE" > "$SERVICE_FILE"

    # Remove empty EXTRA_ENV line if no extra env vars were added
    if [ -z "$EXTRA_ENV" ]; then
        sed -i '/{{EXTRA_ENV}}/d' "$SERVICE_FILE"
    fi

    print_info "Service file created at: $SERVICE_FILE"

    # Reload systemd
    print_info "Reloading systemd daemon..."
    systemctl --user daemon-reload

    # Enable and start service
    read -p "Enable service to start on login? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        systemctl --user enable "${SERVICE_NAME}.service"
        print_info "Service enabled"
    fi

    read -p "Start service now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        systemctl --user start "${SERVICE_NAME}.service"
        print_info "Service started"

        # Show status
        sleep 2
        systemctl --user status "${SERVICE_NAME}.service" --no-pager || true
    fi

    print_info "Setup complete!"
    print_info ""
    print_info "Useful commands:"
    print_info "  Start:   systemctl --user start ${SERVICE_NAME}"
    print_info "  Stop:    systemctl --user stop ${SERVICE_NAME}"
    print_info "  Status:  systemctl --user status ${SERVICE_NAME}"
    print_info "  Logs:    journalctl --user -u ${SERVICE_NAME} -f"
    print_info "  Enable:  systemctl --user enable ${SERVICE_NAME}"
    print_info "  Disable: systemctl --user disable ${SERVICE_NAME}"
}

# Function to remove service
remove_service() {
    read -p "Service name to remove [$SERVICE_NAME]: " CUSTOM_SERVICE_NAME
    SERVICE_NAME="${CUSTOM_SERVICE_NAME:-$SERVICE_NAME}"
    SERVICE_FILE="$USER_SERVICE_DIR/${SERVICE_NAME}.service"

    if [ ! -f "$SERVICE_FILE" ]; then
        print_error "Service file not found: $SERVICE_FILE"
        exit 1
    fi

    print_warn "This will remove the service: $SERVICE_NAME"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi

    # Stop and disable service
    systemctl --user stop "${SERVICE_NAME}.service" 2>/dev/null || true
    systemctl --user disable "${SERVICE_NAME}.service" 2>/dev/null || true

    # Remove service file
    rm -f "$SERVICE_FILE"

    # Reload systemd
    systemctl --user daemon-reload

    print_info "Service removed successfully"
}

# Main menu
main() {
    echo "Claude Code Proxy API - Systemd Service Setup"
    echo "============================================="
    echo
    echo "1) Setup systemd service"
    echo "2) Remove systemd service"
    echo "3) Exit"
    echo
    read -p "Select option (1-3): " -n 1 -r
    echo

    case $REPLY in
        1)
            setup_service
            ;;
        2)
            remove_service
            ;;
        3)
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

# Run main function
main
