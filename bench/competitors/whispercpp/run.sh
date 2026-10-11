#!/bin/bash
# Benchmark protocol for stock whisper.cpp v1.9.5 (ggml-org, commit d1be6fde) with Whisper large-v3-turbo:
# Metal GPU backend (f16 / q8_0 / q5_0), Core ML encoder (ANE) + Metal decoder, and whisper-cli default decoding.
#
#   run.sh                     all variants, full protocol (latency clips x5 + eval-160)
#   run.sh VARIANT...          only these: metal-f16 metal-q8_0 metal-q5_0 coreml-f16 metal-f16-cli-defaults
#   run.sh --smoke [VARIANT..] short.wav only, 2 warm-ups + 3 runs; output goes to smoke/, not bench/results
#
# Env knobs: SKIP_CLI_EVAL=1 (skip eval-160 for metal-f16-cli-defaults), IDLE_WAIT_S=N (max wait for an idle GPU
# before each variant, default 300), COOLDOWN_S=N (idle seconds between
# variants, default 20).
#
# Per variant, one process: decode all audio to 16 kHz mono f32 (ExtAudioFile), load the model once,
# warm up with 4 transcriptions (each latency clip once), then time only whisper_full():
#   bench/results/whispercpp-<variant>-latency.tsv   clip_name \t run_index (1..5) \t ms \t text
#   bench/results/whispercpp-<variant>-eval.tsv      file_name \t ms \t text
# whisper/ggml logs go to bench/competitors/whispercpp/logs/<variant>.log.
# Builds/models come from setup.sh (run once beforehand).
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"   # this harness: bench/competitors/whispercpp
REPO="$(cd "$ROOT/../../.." && pwd)"                    # repo root, 3 levels up
BENCH=$REPO/bench
AUDIO=$BENCH/audio
EVAL_LIST=$BENCH/data/eval-160.tsv
MODELS=$ROOT/models
LOGS=$ROOT/logs
ALL_VARIANTS=(metal-f16 metal-q8_0 metal-q5_0 coreml-f16 metal-f16-cli-defaults)

SMOKE=0
if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi
if [ $# -gt 0 ]; then VARIANTS=("$@"); else VARIANTS=("${ALL_VARIANTS[@]}"); fi

if [ "$SMOKE" = 1 ]; then OUT=$ROOT/smoke; else OUT=$BENCH/results; fi
COOLDOWN_S=${COOLDOWN_S:-20}
mkdir -p "$OUT" "$LOGS"

# Run stock defaults: drop any env overrides that ggml-metal / whisper.cpp read
# (GGML_METAL_TENSOR_DISABLE, GGML_METAL_FUSION_DISABLE, ANEFORGE_ENCODER, ...).
for v in $(env | awk -F= '/^(GGML_|ANEFORGE_|WHISPER_)/ {print $1}'); do unset "$v"; done

for f in "$ROOT/bin/whisper-bench-metal" "$ROOT/bin/whisper-bench-coreml" \
         "$MODELS/ggml-large-v3-turbo.bin" "$MODELS/ggml-large-v3-turbo-q8_0.bin" "$MODELS/ggml-large-v3-turbo-q5_0.bin" \
         "$MODELS/ggml-large-v3-turbo-encoder.mlmodelc/model.mil" "$EVAL_LIST"; do
    [ -e "$f" ] || { echo "missing $f - run $ROOT/setup.sh first" >&2; exit 1; }
done

if [ "$SMOKE" = 1 ]; then
    CLIPS=(--clip "short=$AUDIO/short.wav")
    RUNS=3; WARMUP=2
else
    CLIPS=(--clip "short=$AUDIO/short.wav" --clip "medium=$AUDIO/medium.wav"
           --clip "long=$AUDIO/long.wav"   --clip "xlong=$AUDIO/xlong.wav")
    RUNS=5; WARMUP=4
fi

# GPU "Device Utilization %" from IOAccelerator (no sudo needed). Before each variant, wait up to
# IDLE_WAIT_S (default 300) for 3 consecutive readings < 20% so we don't overlap other GPU work.
gpu_util() {
    local u
    u=$(ioreg -r -c IOAccelerator -d 1 2>/dev/null | sed -n 's/.*"Device Utilization %"=\([0-9]*\).*/\1/p' | sed -n 1p)
    echo "${u:-0}"
}
wait_gpu_idle() {
    local quiet=0 waited=0 u=0
    while [ "$waited" -lt "${IDLE_WAIT_S:-300}" ]; do
        u=$(gpu_util)
        if [ "$u" -lt 20 ]; then quiet=$((quiet + 1)); else quiet=0; fi
        if [ "$quiet" -ge 3 ]; then return 0; fi
        sleep 1; waited=$((waited + 1))
    done
    echo "!! GPU still busy (${u}%) after ${IDLE_WAIT_S:-300}s - timings may overlap other work" | tee -a "$LOGS/runs.txt"
}

{
    echo "== run $(date '+%Y-%m-%d %H:%M:%S')  smoke=$SMOKE  variants: ${VARIANTS[*]}"
    echo "whisper.cpp $(git -C "$ROOT/src" describe --tags 2>/dev/null) $(git -C "$ROOT/src" rev-parse HEAD 2>/dev/null)"
    sw_vers | tr '\n' ' '; echo
    pmset -g | awk '/lowpowermode|powermode/'
    pmset -g batt | sed -n 1p
} | tee -a "$LOGS/runs.txt"

first=1
FAILED=()
for v in "${VARIANTS[@]}"; do
    BIN=$ROOT/bin/whisper-bench-metal
    PRESET=greedy
    EXTRA=()
    EVAL=(--eval-list "$EVAL_LIST" --eval-out "$OUT/whispercpp-$v-eval.tsv")
    case "$v" in
        metal-f16)              MODEL=$MODELS/ggml-large-v3-turbo.bin ;;
        metal-q8_0)             MODEL=$MODELS/ggml-large-v3-turbo-q8_0.bin ;;
        metal-q5_0)             MODEL=$MODELS/ggml-large-v3-turbo-q5_0.bin ;;
        coreml-f16)             MODEL=$MODELS/ggml-large-v3-turbo.bin   # encoder: ggml-large-v3-turbo-encoder.mlmodelc
                                BIN=$ROOT/bin/whisper-bench-coreml
                                EXTRA=(--require-coreml) ;;
        metal-f16-cli-defaults) MODEL=$MODELS/ggml-large-v3-turbo.bin
                                PRESET=cli-defaults
                                if [ "${SKIP_CLI_EVAL:-0}" = 1 ]; then EVAL=(); fi ;;
        *) echo "unknown variant $v (known: ${ALL_VARIANTS[*]})" >&2; exit 1 ;;
    esac
    [ "$SMOKE" = 1 ] && EVAL=()

    if [ "$first" = 0 ] && [ "$SMOKE" = 0 ] && [ "$COOLDOWN_S" -gt 0 ]; then
        echo "-- cooling down ${COOLDOWN_S}s"; sleep "$COOLDOWN_S"
    fi
    first=0
    wait_gpu_idle

    if [ "$SMOKE" = 1 ]; then LOG=$LOGS/smoke-$v; else LOG=$LOGS/$v; fi
    echo "== $v  ($(date '+%H:%M:%S'))  model=$(basename "$MODEL") preset=$PRESET bin=$(basename "$BIN") gpu_util_before=$(gpu_util)%" | tee -a "$LOGS/runs.txt"
    if ! "$BIN" --label "$v" --model "$MODEL" --preset "$PRESET" --threads 4 \
            "${CLIPS[@]}" --runs "$RUNS" --warmup "$WARMUP" \
            --latency-out "$OUT/whispercpp-$v-latency.tsv" \
            ${EVAL[@]+"${EVAL[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} 2> "$LOG.log" | tee "$LOG.out"; then
        echo "!! $v FAILED - see $LOG.log" | tee -a "$LOGS/runs.txt"
        FAILED+=("$v")
    fi
done
echo "== done $(date '+%H:%M:%S'); results in $OUT" | tee -a "$LOGS/runs.txt"
if [ ${#FAILED[@]} -gt 0 ]; then echo "failed variants: ${FAILED[*]}" >&2; exit 1; fi
