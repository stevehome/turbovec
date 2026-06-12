#!/usr/bin/env bash
# Extracts ANTHROPIC_API_KEY from the project .env and appends it to ~/.zshrc.

ENV_FILE="$(dirname "$0")/../.env"
PROFILE="$HOME/.zshrc"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: .env not found at $ENV_FILE" >&2
  exit 1
fi

KEY=$(grep '^ANTHROPIC_API_KEY=' "$ENV_FILE" | cut -d '=' -f2-)

if [ -z "$KEY" ]; then
  echo "Error: ANTHROPIC_API_KEY not found in $ENV_FILE" >&2
  exit 1
fi

if grep -q 'ANTHROPIC_API_KEY' "$PROFILE"; then
  echo "ANTHROPIC_API_KEY already present in $PROFILE — skipping."
else
  echo "export ANTHROPIC_API_KEY=\"$KEY\"" >> "$PROFILE"
  echo "Added ANTHROPIC_API_KEY to $PROFILE"
  echo "Run: source $PROFILE"
fi
