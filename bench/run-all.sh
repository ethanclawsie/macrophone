#!/usr/bin/env bash
# Runs every Whisper large-v3-turbo implementation through the same protocol, one at a time,
# then scores them all with the same WER normalizer.
#
#   bench/run-all.sh            everything (~30 min)
#   bench/run-all.sh macrophone      only Macrophone's variants
#
# Protocol (identical for every harness): load the model once, decode audio to 16 kHz float32
# before timing, warm up, then time only the transcription of in-memory audio.
#   latency: bench/audio/{short,medium,long,xlong}.wav, 5 runs each
#   accuracy: bench/data/eval-160.tsv (LibriSpeech test-clean, 40 per length bucket), 1 run each
set -uo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/bench/results/run-all.log"
MODELS="$HOME/Library/Application Support/Macrophone/models"
COOLDOWN_S=${COOLDOWN_S:-20}
mkdir -p bench/results

step() {
    echo "=== $(date '+%H:%M:%S') $*" | tee -a "$LOG"
    "$@" >>"$LOG" 2>&1 || echo "!!! failed: $*" | tee -a "$LOG"
    sleep "$COOLDOWN_S"
}

run_macrophone() {
    swift build -c release --product macrophone-bench >/dev/null 2>&1
    step .build/release/macrophone-bench protocol --name macrophone --model "$MODELS/ggml-large-v3-turbo-q4_0.bin"
    step .build/release/macrophone-bench protocol --name macrophone-q8 --model "$MODELS/ggml-large-v3-turbo-q8_0.bin"
}

if [ "${1:-}" = "macrophone" ]; then
    run_macrophone
else
    run_macrophone
    step bench/competitors/whispercpp/run.sh
    step bench/competitors/whisperkit/run.sh
    step bench/competitors/mlx/run.sh
    step bench/competitors/longtail/run.sh
fi

echo "=== $(date '+%H:%M:%S') scoring" | tee -a "$LOG"
# WER with Whisper's English text normalizer (the Open ASR Leaderboard's) + median latency on the eval set
uv run --quiet --no-project --with whisper-normalizer==0.1.15 python -I bench/score.py | tee bench/results/wer.txt
# median latency of each latency clip
.build/release/macrophone-bench table | tee bench/results/latency-table.md
