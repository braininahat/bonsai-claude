#!/usr/bin/env bash
# Install mlx-claude onto your PATH (symlink into ~/.local/bin).
#
# Usage:
#   ./install.sh
#
# Custom install dir:
#   MLX_CLAUDE_BIN_DIR=/usr/local/bin ./install.sh
#
# Uninstall:
#   ./uninstall.sh

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SCRIPT="$REPO_DIR/mlx-claude.py"
BIN_DIR="${MLX_CLAUDE_BIN_DIR:-$HOME/.local/bin}"
LINK="$BIN_DIR/mlx-claude"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is required. Install with:" >&2
  echo "       curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "WARNING: 'claude' CLI not on PATH. mlx-claude needs it to launch." >&2
  echo "         Setup: https://docs.claude.com/en/docs/claude-code/setup" >&2
fi

mkdir -p "$BIN_DIR"
chmod +x "$SCRIPT"
ln -sfn "$SCRIPT" "$LINK"

echo "Installed: $LINK -> $SCRIPT"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "NOTE: $BIN_DIR is not on your PATH."
    echo "Add to your shell rc (~/.zshrc, ~/.bashrc):"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac

echo
echo "Run: mlx-claude"
