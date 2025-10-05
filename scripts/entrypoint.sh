#!/usr/bin/env bash
#
# Docker Entrypoint Script for configurable UID/GID containers.
#
# This script creates (or reuses) a user and group inside the container so that
# mounted volumes remain accessible when mapped to a host UID/GID. It supports
# both rootless execution (no user management) and privileged execution where it
# manages the user before dropping privileges with setpriv.

set -euo pipefail

APT_PACKAGE="${APT_PACKAGE:-}"
if [[ -n "${APT_PACKAGE}" ]]; then
  echo "APT_PACKAGE specified: ${APT_PACKAGE}"
  packages=$(echo "${APT_PACKAGE}" | tr ',' ' ')
  echo "Installing packages: ${packages}"
  apt-get update && apt-get install -y ${packages} && rm -rf /var/lib/apt/lists/*
fi

run_init_script_current() {
  local script_path="$1"
  if [[ -z "${script_path}" ]]; then
    return
  fi
  if [[ ! -f "${script_path}" ]]; then
    echo "Init script ${script_path} not found; skipping" >&2
    return
  fi
  if [[ -x "${script_path}" ]]; then
    "${script_path}"
  else
    bash "${script_path}"
  fi
}

run_init_script_as_user() {
  local script_path="$1"
  local uid="$2"
  local gid="$3"
  if [[ -z "${script_path}" ]]; then
    return
  fi
  if [[ ! -f "${script_path}" ]]; then
    echo "Init script ${script_path} not found; skipping" >&2
    return
  fi
  local runner=(setpriv "--reuid=${uid}" "--regid=${gid}" --init-groups)
  if [[ -x "${script_path}" ]]; then
    "${runner[@]}" "${script_path}"
  else
    "${runner[@]}" bash "${script_path}"
  fi
}

should_enable_sudo() {
  local value="${ENABLE_SUDO:-}"
  if [[ -z "${value}" ]]; then
    return 1
  fi
  value="${value,,}"
  case "${value}" in
  1 | true | yes | on)
    return 0
    ;;
  *)
    return 1
    ;;
  esac
}

APP_USER="${APP_USER:-appuser}"
APP_GROUP="${APP_GROUP:-appgroup}"
USER_HOME="${USER_HOME:-/home/user}" # Default home if we need to create the user
WORKDIR="${WORKDIR:-/workspace}"
PUID="${PUID:-1000}"
PGID="${PGID:-1000}"

# When the container runs without root privileges, we cannot manage system
# users. Simply export HOME and execute the command.
if [[ "${EUID}" -ne 0 ]]; then
  export HOME="${USER_HOME}"
  export USER="${APP_USER}"
  cd "${WORKDIR}"
  if should_enable_sudo; then
    echo "ENABLE_SUDO requested but container is not running as root; skipping" >&2
  fi
  init_script="${INIT_SCRIPT:-}"
  if [[ -n "${init_script}" ]]; then
    echo "Running init script as ${USER}: ${init_script}"
    run_init_script_current "${init_script}"
  fi
  echo "Not running as root, executing command directly: $*"
  exec "$@"
fi

echo "Running as root, configuring container user"
echo "Requested UID=${PUID}, GID=${PGID}"
echo "Target user=${APP_USER}, group=${APP_GROUP}"
echo "USER_HOME=${USER_HOME}"
echo "WORKDIR=${WORKDIR}"

user_exists() {
  id "$1" &>/dev/null
}

group_exists() {
  getent group "$1" &>/dev/null
}

# Determine the actual group name backed by PGID, creating or updating as needed.
actual_group="${APP_GROUP}"
if group_exists "${APP_GROUP}"; then
  current_gid=$(getent group "${APP_GROUP}" | cut -d: -f3)
  if [[ "${current_gid}" != "${PGID}" ]]; then
    if getent group "${PGID}" &>/dev/null; then
      existing_group=$(getent group "${PGID}" | cut -d: -f1)
      echo "Group ${APP_GROUP} currently uses GID ${current_gid}"
      echo "Using existing group ${existing_group} for requested GID ${PGID}"
      actual_group="${existing_group}"
    else
      echo "Updating ${APP_GROUP} group GID ${current_gid} -> ${PGID}"
      groupmod -g "${PGID}" "${APP_GROUP}"
    fi
  fi
else
  if getent group "${PGID}" &>/dev/null; then
    existing_group=$(getent group "${PGID}" | cut -d: -f1)
    echo "GID ${PGID} already taken; using existing group ${existing_group}"
    actual_group="${existing_group}"
  else
    echo "Creating group ${APP_GROUP} with GID ${PGID}"
    groupadd -g "${PGID}" "${APP_GROUP}"
  fi
fi

APP_GROUP="${actual_group}"

# Determine the actual user name backed by PUID, creating or updating as needed.
if user_exists "${APP_USER}"; then
  current_uid=$(id -u "${APP_USER}")
  current_gid=$(id -g "${APP_USER}")
  echo "User ${APP_USER} exists (UID=${current_uid}, GID=${current_gid})"

  if [[ "${current_uid}" != "${PUID}" ]]; then
    if getent passwd "${PUID}" &>/dev/null; then
      conflict_user=$(getent passwd "${PUID}" | cut -d: -f1)
      if [[ "${conflict_user}" != "${APP_USER}" ]]; then
        echo "Warning: UID ${PUID} already used by ${conflict_user}"
        echo "Keeping existing UID ${current_uid} for ${APP_USER}"
        PUID="${current_uid}"
      else
        echo "Updating ${APP_USER} UID ${current_uid} -> ${PUID}"
        usermod -u "${PUID}" "${APP_USER}"
      fi
    else
      usermod -u "${PUID}" "${APP_USER}"
    fi
  fi

  if [[ "${current_gid}" != "${PGID}" ]]; then
    echo "Updating ${APP_USER} primary group to ${APP_GROUP}"
    usermod -g "${APP_GROUP}" "${APP_USER}"
  fi
else
  if getent passwd "${PUID}" &>/dev/null; then
    existing_user=$(getent passwd "${PUID}" | cut -d: -f1)
    echo "Warning: UID ${PUID} already used by ${existing_user}"
    echo "Searching for a free UID"
    next_uid=1001
    while getent passwd "${next_uid}" &>/dev/null; do
      ((next_uid++))
    done
    echo "Assigning UID ${next_uid} to ${APP_USER}"
    PUID="${next_uid}"
  fi

  echo "Creating user ${APP_USER} (UID=${PUID}, GID=${PGID})"
  useradd -u "${PUID}" -g "${APP_GROUP}" -d "${USER_HOME}" -s /bin/bash -m "${APP_USER}"
fi

# Refresh user metadata after potential modifications.
FINAL_UID=$(id -u "${APP_USER}")
FINAL_GID=$(id -g "${APP_USER}")
actual_home=$(getent passwd "${APP_USER}" | cut -d: -f6)
if [[ -n "${actual_home}" ]]; then
  USER_HOME="${actual_home}"
fi

# Ensure the home directory exists and is owned correctly.
mkdir -p "${USER_HOME}"
chown -R "${APP_USER}:${APP_GROUP}" "${USER_HOME}" || true
mkdir -p "${USER_HOME}/.cache" "${USER_HOME}/.config" "${USER_HOME}/.local"

# Ensure the workspace exists and is owned correctly.
mkdir -p "${WORKDIR}"
chown -R "${APP_USER}:${APP_GROUP}" "${WORKDIR}" || true

if should_enable_sudo; then
  if command -v sudo >/dev/null 2>&1; then
    echo "Enabling passwordless sudo for ${APP_USER}"
    if getent group sudo >/dev/null 2>&1; then
      usermod -aG sudo "${APP_USER}"
    fi
    echo "${APP_USER} ALL=(ALL) NOPASSWD:ALL" >"/etc/sudoers.d/${APP_USER}"
    chmod 0440 "/etc/sudoers.d/${APP_USER}"
  else
    echo "ENABLE_SUDO requested but sudo is not installed; skipping" >&2
  fi
fi

export USER="${APP_USER}"
export HOME="${USER_HOME}"
export WORKDIR="${WORKDIR}"

cd "${WORKDIR}"

echo "PUID/PGID configuration complete"
echo "  Requested UID=${PUID}, GID=${PGID}"
echo "  Final UID=${FINAL_UID}, GID=${FINAL_GID}"
echo "  User=${APP_USER}, Group=${APP_GROUP}"
echo "  Home=${USER_HOME}"
echo "  Workspace=${WORKDIR}"

init_script="${INIT_SCRIPT:-}"
if [[ -n "${init_script}" ]]; then
  echo "Running init script as ${APP_USER}: ${init_script}"
  run_init_script_as_user "${init_script}" "${FINAL_UID}" "${FINAL_GID}"
fi

echo "Starting application: $*"
exec setpriv --reuid="${FINAL_UID}" --regid="${FINAL_GID}" --init-groups "$@"
