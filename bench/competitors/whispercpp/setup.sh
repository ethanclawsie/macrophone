#!/bin/bash
# One-time setup for the stock whisper.cpp competitor benchmark (does NOT run the benchmark).
#   - clones ggml-org/whisper.cpp at tag v1.9.5 into src/ (pristine, untouched afterwards)
#   - builds two static Release trees: build-metal (Metal) and build-coreml (Core ML encoder + Metal)
#   - compiles the harness (harness/whisper_bench.cpp) against each tree into bin/
#   - unzips the Core ML encoder and symlinks the read-only ggml models into models/
#   - optionally (--ane-compile) loads the Core ML model once so the ANE compile is cached
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"   # this harness: bench/competitors/whispercpp
SRC=$ROOT/src
MODELS_SRC="$HOME/Library/Application Support/Macrophone/models"   # shared with the app (scripts/fetch-models.sh)
TAG=v1.9.5

if [ ! -d "$SRC/.git" ]; then
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp "$SRC"
fi
echo "whisper.cpp: $(git -C "$SRC" describe --tags) $(git -C "$SRC" rev-parse HEAD)"

COMMON_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DWHISPER_BUILD_TESTS=OFF
    -DWHISPER_BUILD_SERVER=OFF
    -DWHISPER_BUILD_EXAMPLES=ON
)

cmake -S "$SRC" -B "$ROOT/build-metal"  "${COMMON_FLAGS[@]}"
cmake --build "$ROOT/build-metal" --target whisper whisper-cli

cmake -S "$SRC" -B "$ROOT/build-coreml" "${COMMON_FLAGS[@]}" -DWHISPER_COREML=ON -DWHISPER_COREML_ALLOW_FALLBACK=ON
cmake --build "$ROOT/build-coreml" --target whisper whisper-cli

# ---- harness ----
mkdir -p "$ROOT/bin"
build_harness() {  # $1 = build dir, $2 = output binary, $3... = extra libs/frameworks
    local B=$1 OUT=$2; shift 2
    clang++ -std=c++17 -O2 -Wall -Wextra \
        -I"$SRC/include" -I"$SRC/ggml/include" \
        "$ROOT/harness/whisper_bench.cpp" \
        "$B/src/libwhisper.a" "$@" \
        "$B/ggml/src/libggml.a" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/ggml-blas/libggml-blas.a" \
        "$B/ggml/src/ggml-metal/libggml-metal.a" "$B/ggml/src/libggml-base.a" \
        -framework Accelerate -framework Metal -framework MetalKit -framework Foundation \
        -framework AudioToolbox -framework CoreFoundation \
        -o "$OUT"
}
build_harness "$ROOT/build-metal"  "$ROOT/bin/whisper-bench-metal"
build_harness "$ROOT/build-coreml" "$ROOT/bin/whisper-bench-coreml" "$ROOT/build-coreml/src/libwhisper.coreml.a" -framework CoreML

# ---- models ----
# whisper.cpp (src/whisper.cpp, whisper_get_coreml_path_encoder) looks for the Core ML encoder
# next to the .bin: strip ".bin", strip a trailing "-qX_Y" quant suffix, append "-encoder.mlmodelc".
mkdir -p "$ROOT/models"
for f in ggml-large-v3-turbo.bin ggml-large-v3-turbo-q8_0.bin ggml-large-v3-turbo-q5_0.bin; do
    ln -sfn "$MODELS_SRC/$f" "$ROOT/models/$f"
done
if [ ! -d "$ROOT/models/ggml-large-v3-turbo-encoder.mlmodelc" ]; then
    unzip -q "$MODELS_SRC/ggml-large-v3-turbo-encoder.mlmodelc.zip" -x '__MACOSX/*' -d "$ROOT/models"
fi

# ---- ANE first compile (load only, no transcription) ----
if [ "${1:-}" = "--ane-compile" ]; then
    mkdir -p "$ROOT/logs"
    /usr/bin/time -p "$ROOT/bin/whisper-bench-coreml" --model "$ROOT/models/ggml-large-v3-turbo.bin" \
        --warmup 0 --require-coreml --label ane-compile 2> "$ROOT/logs/ane-compile.log"
fi
echo "setup done"
