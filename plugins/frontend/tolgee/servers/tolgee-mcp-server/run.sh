#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..." >&2
  npm install >&2
fi

# Build if dist doesn't exist or if src is newer than dist
if [ ! -d "dist" ] || [ "$(find src -name '*.ts' -newer dist/index.js 2>/dev/null | head -1)" ]; then
  echo "Building..." >&2
  npm run build >&2
fi

# Run the server
exec node dist/index.js "$@"
