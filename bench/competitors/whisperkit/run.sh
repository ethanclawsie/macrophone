#!/bin/bash
# WhisperKit (argmax-oss-swift v1.1.1) running Whisper large-v3-turbo Core ML models.
# Runs the full protocol for every variant when called with no arguments:
#   load once -> decode audio to 16 kHz mono floats -> 2 warmups (xlong.wav)
#   -> latency: short/medium/long/xlong x 5 runs -> accuracy: eval-160, each utterance once.
# Results: bench/results/<variant>-latency.tsv and <variant>-eval.tsv (relative to the repo root)
#
# Usage: run.sh [variant ...]
#   default variants: whisperkit whisperkit-gpudec
#   opt-in variant:   whisperkit-632mb (compressed turbo; measured speed-neutral vs default, so not run by default)
#   SMOKE=1 run.sh    quick check: short.wav x 3 only, no eval, written to $HERE/smoke/ instead of bench/results
#
# One-time setup, from this directory, after bench/make-audio.sh and bench/fetch-data.sh have built
# bench/audio/*.wav and bench/data/eval-160.tsv.
# The download pins the Core ML model and tokenizer revisions (see Bench.swift) and saves them to ./hf:
#   swift build -c release
#   .build/release/whisperkit-bench download --hf-base "$PWD/hf" --variant openai_whisper-large-v3-v20240930_turbo        # 1.5 GB
#   .build/release/whisperkit-bench download --hf-base "$PWD/hf" --variant openai_whisper-large-v3-v20240930_turbo_632MB  # opt-in, 620 MB
set -euo pipefail

HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"   # bench/competitors/whisperkit is three levels below the repo root
BIN=$HERE/.build/release/whisperkit-bench
HF=$HERE/hf
MODELS=$HF/models/argmaxinc/whisperkit-coreml
AUDIO=$REPO/bench/audio
EVAL=$REPO/bench/data/eval-160.tsv
RESULTS=$REPO/bench/results
LOGS=$HERE/logs

DEFAULT_VARIANTS=(whisperkit whisperkit-gpudec)

# label -> "model-folder mel-units encoder-units decoder-units"   (units: cpu | gpu | ane | all)
variant_config() {
  case "$1" in
    # WhisperKit default ModelComputeOptions: mel GPU, encoder Neural Engine, decoder Neural Engine.
    whisperkit)        echo "openai_whisper-large-v3-v20240930_turbo gpu ane ane" ;;
    # Same model, text decoder on the GPU (faster per decoder step on M5 in setup measurements).
    whisperkit-gpudec) echo "openai_whisper-large-v3-v20240930_turbo gpu ane gpu" ;;
    # Opt-in: 4/6-bit palettized turbo, default compute units.
    whisperkit-632mb)  echo "openai_whisper-large-v3-v20240930_turbo_632MB gpu ane ane" ;;
    *) return 1 ;;
  esac
}

# Reuse the already-built binary (its name keys the warm Core ML compile cache); build only if missing.
if [ ! -x "$BIN" ]; then
  swift build -c release --package-path "$HERE"
fi

if [ "${SMOKE:-0}" = "1" ]; then
  OUT=$HERE/smoke
  PROTOCOL=(--clips short --runs 3)
else
  OUT=$RESULTS
  PROTOCOL=(--clips short,medium,long,xlong --runs 5 --warmup-clip "$AUDIO/xlong.wav")
fi
mkdir -p "$OUT" "$LOGS"

if [ $# -gt 0 ]; then VARIANTS=("$@"); else VARIANTS=("${DEFAULT_VARIANTS[@]}"); fi

for label in "${VARIANTS[@]}"; do
  config=$(variant_config "$label") || { echo "unknown variant: $label" >&2; exit 2; }
  read -r model mel encoder decoder <<<"$config"
  EVAL_ARGS=(--eval-tsv "$EVAL" --eval-out "$OUT/$label-eval.tsv")
  [ "${SMOKE:-0}" = "1" ] && EVAL_ARGS=()
  echo "=== $label: $model (mel=$mel encoder=$encoder decoder=$decoder) $(date '+%F %T')"
  "$BIN" bench \
    --label "$label" \
    --model "$MODELS/$model" \
    --tokenizer-base "$HF" \
    --mel "$mel" --encoder "$encoder" --decoder "$decoder" \
    --warmup 2 \
    --audio-dir "$AUDIO" "${PROTOCOL[@]}" \
    --latency-out "$OUT/$label-latency.tsv" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    2>&1 | tee "$LOGS/run-$label$([ "${SMOKE:-0}" = "1" ] && echo -smoke).log"
done
