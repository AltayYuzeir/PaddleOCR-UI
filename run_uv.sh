#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install from https://astral.sh/uv" >&2
  exit 1
fi

# Ensure a .venv exists and is seeded with pip
uv venv --seed

# Fallback: if pip still missing for any reason, try ensurepip
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  .venv/bin/python -m ensurepip --upgrade || true
fi

# Create or reuse .venv, then sync test group
uv sync --group test

# Quick sanity print
uv run python -V
echo "Env ready. Use: 'uv run <cmd>' or '.venv/bin/<cmd>'." 
