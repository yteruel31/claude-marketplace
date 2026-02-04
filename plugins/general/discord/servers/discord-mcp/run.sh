#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
  echo "Error: uv is not installed. Please install it first:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# Capture the original working directory before changing to script dir
# This is the user's project directory where Claude Code was invoked
export ORIGINAL_WORKING_DIR="${PWD}"

# Change to the script's directory (for pyproject.toml)
cd "$(dirname "$0")"

# Run the server using uv (auto-manages venv and dependencies from pyproject.toml)
exec uv run python discord_mcp.py "$@"
