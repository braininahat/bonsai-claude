#!/usr/bin/env bash
# Remove the mlx-claude symlink from ~/.local/bin (or MLX_CLAUDE_BIN_DIR).

set -euo pipefail

BIN_DIR="${MLX_CLAUDE_BIN_DIR:-$HOME/.local/bin}"
LINK="$BIN_DIR/mlx-claude"

if [[ -L "$LINK" || -e "$LINK" ]]; then
  rm -f "$LINK"
  echo "Removed: $LINK"
else
  echo "Nothing to remove: $LINK (not present)"
fi
