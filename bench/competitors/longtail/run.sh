#!/bin/bash
# Long-tail Whisper large-v3-turbo runtimes: full latency + accuracy protocol.
#
#   ./setup.sh                        # once: creates venv-hf and venv-fw at the pinned versions
#   ./run.sh                          # hf-mps, then fasterwhisper-cpu
#   ./run.sh fasterwhisper-cpu        # just one runtime
#   ./run.sh --smoke [runtime ...]    # short+medium x2 only, written to $SMOKE_DIR
#
# Writes bench/results/<name>-latency.tsv (clip_name, run_index, ms, text) and
# bench/results/<name>-eval.tsv (file_name, ms, text). Each runtime loads once,
# decodes all audio up front, warms up on each latency clip once (as the mlx and
# whisper.cpp harnesses do), then times only the transcription call.
# Don't run this while other benchmarks are timing.
#
#   hf-mps             transformers + PyTorch MPS, fp16, openai/whisper-large-v3-turbo
#   fasterwhisper-cpu  faster-whisper / CTranslate2 on CPU, mobiuslabsgmbh/faster-whisper-large-v3-turbo
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"  # bench/competitors/longtail is 3 levels below the repo root
RESULTS="$REPO/bench/results"

# Settings picked from smoke tests; override via env.
#   HF_ATTN: sdpa | eager | eager-encoder (eager encoder self-attention + sdpa decoder;
#            torch 2.11's MPS sdpa is ~2.5x slower than eager on the 1500-token encoder)
HF_ATTN="${HF_ATTN:-eager-encoder}"
#   FW_COMPUTE: float32 | int8 | int8_float32. This CTranslate2 wheel has no OpenMP: float32
#            GEMMs go to Accelerate (AMX/SME), int8 to Ruy. Smoke test, medium.wav:
#            float32 ~2.4 s, int8 ~3.5 s with 10 threads, ~5.3 s with 4.
FW_COMPUTE="${FW_COMPUTE:-float32}"
FW_THREADS="${FW_THREADS:-4}"       # cpu_threads (P-cores); Accelerate threads its own GEMMs

export HF_HOME="$SCRIPT_DIR/hf_home" HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
# No network during runs once both weight snapshots are cached.
if [ -d "$HF_HOME/hub/models--openai--whisper-large-v3-turbo/snapshots" ] &&
   [ -d "$HF_HOME/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots" ]; then
  export HF_HUB_OFFLINE=1
fi

smoke=0
if [ "${1:-}" = "--smoke" ]; then smoke=1; shift; fi
runtimes=("$@")
[ ${#runtimes[@]} -eq 0 ] && runtimes=(hf-mps fasterwhisper-cpu)

# The venvs come from setup.sh (exact pinned versions). Check every requested one
# before anything runs, so a missing venv doesn't stop a run halfway through.
for rt in "${runtimes[@]}"; do
  case "$rt" in
    hf-mps)            venv="$SCRIPT_DIR/venv-hf" ;;
    fasterwhisper-cpu) venv="$SCRIPT_DIR/venv-fw" ;;
    *) echo "unknown runtime: $rt (expected hf-mps or fasterwhisper-cpu)" >&2; exit 2 ;;
  esac
  if [ ! -x "$venv/bin/python" ]; then
    echo "$venv is missing: run $SCRIPT_DIR/setup.sh first" >&2
    exit 1
  fi
done

for rt in "${runtimes[@]}"; do
  case "$rt" in
    hf-mps)            py="$SCRIPT_DIR/venv-hf/bin/python"; opts=(--attn "$HF_ATTN") ;;
    fasterwhisper-cpu) py="$SCRIPT_DIR/venv-fw/bin/python"; opts=(--compute-type "$FW_COMPUTE" --threads "$FW_THREADS") ;;
    *) echo "unknown runtime: $rt (expected hf-mps or fasterwhisper-cpu)" >&2; exit 2 ;;
  esac
  echo "== $rt ${opts[*]}" >&2
  if [ "$smoke" -eq 1 ]; then
    SMOKE_DIR="${SMOKE_DIR:-$(mktemp -d)}"
    "$py" -I "$SCRIPT_DIR/bench_longtail.py" "$rt" "${opts[@]}" --clips short,medium --runs 2 \
      --latency-out "$SMOKE_DIR/$rt-latency.tsv"
  else
    mkdir -p "$RESULTS"
    "$py" -I "$SCRIPT_DIR/bench_longtail.py" "$rt" "${opts[@]}" \
      --latency-out "$RESULTS/$rt-latency.tsv" --eval-out "$RESULTS/$rt-eval.tsv"
  fi
done
