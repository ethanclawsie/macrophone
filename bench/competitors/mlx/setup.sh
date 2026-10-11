#!/usr/bin/env bash
# One-time setup, idempotent, needs network: the uv venv from uv.lock (CPython 3.14, mlx 0.32.3
# macOS-26 wheels, mlx-whisper 0.4.3) and the pinned model snapshots into ./hf (HF_HOME).
set -euo pipefail

# This harness's directory (bench/competitors/mlx), wherever the repo is cloned.
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
uv sync --project "$ROOT" --frozen
HF_HOME=$ROOT/hf "$ROOT/.venv/bin/python" -I "$ROOT/bench.py" --fetch
