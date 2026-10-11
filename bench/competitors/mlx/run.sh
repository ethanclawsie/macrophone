#!/usr/bin/env bash
# MLX Whisper (mlx-whisper 0.4.3 on MLX 0.32.3), Whisper large-v3-turbo: full benchmark protocol.
#
# Each variant runs in its own process: load the model once, warm up on all four latency clips,
# then time transcribe() on in-memory 16 kHz float32 arrays:
#   bench/audio/{short,medium,long,xlong}.wav x 5 runs  -> bench/results/<variant>-latency.tsv
#   bench/data/eval-160.tsv, one run per file           -> bench/results/<variant>-eval.tsv
# Variants: mlx (fp16), mlx-8bit, mlx-4bit. Pass variant names to run only those.
# One-time setup (venv + model download): bench/competitors/mlx/setup.sh
set -euo pipefail

# This harness's directory (bench/competitors/mlx), wherever the repo is cloned. bench.py finds
# bench/audio, bench/data and bench/results relative to its own location.
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY=$ROOT/.venv/bin/python
export HF_HOME=$ROOT/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

if [ $# -gt 0 ]; then VARIANTS=("$@"); else VARIANTS=(mlx mlx-8bit mlx-4bit); fi
[ -x "$PY" ] || { echo "no venv at $ROOT/.venv: run $ROOT/setup.sh first" >&2; exit 1; }

mkdir -p "$ROOT/logs"
LOG=$ROOT/logs/run-$(date +%Y%m%d-%H%M%S).log
echo "log: $LOG"
for v in "${VARIANTS[@]}"; do
    caffeinate -i "$PY" -I "$ROOT/bench.py" --variant "$v" --latency --eval 2>&1 | tee -a "$LOG"
done
