#!/bin/bash
set -e

CLAUDE_HOME=${CLAUDE_HOME:-"/data/home"}
CLAUDE_WORKSPACE=${CLAUDE_WORKSPACE:$CLAUDE_HOME}

# Get PUID and PGID from environment or use defaults
PUID=${PUID:-1000}
PGID=${PGID:-1000}

echo "Starting Claude Proxy with PUID=$PUID, PGID=$PGID"
echo "CLAUDE_HOME=$CLAUDE_HOME"
echo "CLAUDE_WORKSPACE=$CLAUDE_WORKSPACE"

# Function to check if user/group exists
user_exists() {
  id "$1" &>/dev/null
}

group_exists() {
  getent group "$1" &>/dev/null
}

# Handle claude group creation/modification
if group_exists claude; then
  current_gid=$(getent group claude | cut -d: -f3)
  if [[ "$current_gid" != "$PGID" ]]; then
    echo "Claude group exists with GID $current_gid, need to change to $PGID"
    # Check if target GID is already in use by another group
    if getent group "$PGID" &>/dev/null; then
      target_group=$(getent group "$PGID" | cut -d: -f1)
      echo "Warning: GID $PGID is already used by group '$target_group'"
      echo "Removing claude group and adding claude user to existing group '$target_group'"
      groupdel claude || true
      CLAUDE_GROUP_NAME="$target_group"
    else
      echo "Modifying claude group to GID $PGID"
      groupmod -g "$PGID" claude
      CLAUDE_GROUP_NAME="claude"
    fi
  else
    echo "Claude group already has correct GID $PGID"
    CLAUDE_GROUP_NAME="claude"
  fi
else
  # Check if target GID is already in use
  if getent group "$PGID" &>/dev/null; then
    target_group=$(getent group "$PGID" | cut -d: -f1)
    echo "GID $PGID is already used by group '$target_group', will use existing group"
    CLAUDE_GROUP_NAME="$target_group"
  else
    echo "Creating claude group with GID $PGID"
    groupadd -g "$PGID" claude
    CLAUDE_GROUP_NAME="claude"
  fi
fi

# Create or modify claude user
if user_exists claude; then
  current_uid=$(id -u claude)
  current_gid=$(id -g claude)
  echo "Claude user exists with UID $current_uid, GID $current_gid"

  if [[ "$current_uid" != "$PUID" ]]; then
    # Check if target UID is already in use
    if getent passwd "$PUIdoD" &>/dev/null; then
      existing_user=$(getent passwd "$PUID" | cut -d: -f1)
      if [[ "$existing_user" != "claude" ]]; then
        echo "Warning: UID $PUID is already used by user '$existing_user'"
        echo "Cannot modify claude user UID, will use existing UID $current_uid"
        PUID="$current_uid"
      fi
    else
      echo "Modifying claude user UID to $PUID"
      usermod -u "$PUID" claude
    fi
  fi

  # Update group membership and shell
  echo "Setting claude user group to $CLAUDE_GROUP_NAME and shell to /bin/bash"
  usermod -g "$CLAUDE_GROUP_NAME" -s /bin/bash claude
else
  # Check if target UID is already in use
  if getent passwd "$PUID" &>/dev/null; then
    existing_user=$(getent passwd "$PUID" | cut -d: -f1)
    echo "Warning: UID $PUID is already used by user '$existing_user'"
    echo "Will create claude user with a different UID"
    # Find next available UID starting from 1001
    PUID=1001
    while getent passwd "$PUID" &>/dev/null; do
      ((PUID++))
    done
    echo "Using available UID $PUID for claude user"
  fi

  echo "Creating claude user with UID $PUID, group $CLAUDE_GROUP_NAME"
  useradd -u "$PUID" -g "$CLAUDE_GROUP_NAME" -d "$CLAUDE_HOME" -s /bin/bash -m claude
fi

# Ensure claude home directory exists and has correct ownership
echo "Setting up Claude home directory: $CLAUDE_HOME"
mkdir -p "$CLAUDE_HOME"

# Also ensure the user's actual home directory exists with proper ownership
CLAUDE_USER_HOME=$(getent passwd claude | cut -d: -f6)
if [[ "$CLAUDE_USER_HOME" != "$CLAUDE_HOME" ]]; then
  echo "Setting up Claude user home directory: $CLAUDE_USER_HOME"
  mkdir -p "$CLAUDE_USER_HOME"
  chown -R claude:"$CLAUDE_GROUP_NAME" "$CLAUDE_USER_HOME"
fi

# Create additional directories that Claude might need
mkdir -p "$CLAUDE_HOME"/{.cache,.config,.local}
chown -R claude:"$CLAUDE_GROUP_NAME" "$CLAUDE_HOME"

# Ensure workspace directory exists
mkdir -p "$CLAUDE_WORKSPACE"
chown -R claude:"$CLAUDE_GROUP_NAME" "$CLAUDE_WORKSPACE"

# Update environment variables for the application
export CLAUDE_USER="claude"
export CLAUDE_GROUP="$CLAUDE_GROUP_NAME"
export CLAUDE_WORKING_DIRECTORY="$CLAUDE_WORKSPACE"

# Get final UID/GID values
FINAL_PUID=$(id -u claude)
FINAL_PGID=$(id -g claude)

echo "PUID/PGID configuration complete"
echo "  Requested: UID=$PUID, GID=$PGID"
echo "  Final: UID=$FINAL_PUID, GID=$FINAL_PGID"
echo "  User: claude, Group: $CLAUDE_GROUP_NAME"
echo "Enabling privilege dropping to claude user"

echo "Final configuration:"
echo "  Claude user: $(id claude)"
echo "  Claude home: $CLAUDE_HOME ($(ls -ld "$CLAUDE_HOME"))"
echo "  Environment: CLAUDE_USER=$CLAUDE_USER, CLAUDE_GROUP=$CLAUDE_GROUP_NAME"

# Execute the main command
echo "Starting application: $*"
exec "$@"
