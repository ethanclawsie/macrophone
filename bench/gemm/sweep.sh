#!/usr/bin/env bash
# Sweeps ggml-metal's tensor-API matmul tile configuration (M5 Neural Accelerator path) on the
# Whisper encoder's GEMM shapes. Each config rebuilds ggml-metal and the micro-benchmark.
#
#   bench/gemm/sweep.sh [q8_0|q4_0|f16]
#
# Config = N_MM_NK N_MM_BLOCK_X N_MM_BLOCK_Y N_MM_SIMD_GROUP_X N_MM_SIMD_GROUP_Y
#   output tile = (16*BLOCK_Y*SG_Y weight rows) x (16*BLOCK_X*SG_X activation columns), K tile = 16*NK
set -uo pipefail
CDPATH= cd -- "$(dirname -- "$0")/../.."
TYPE=${1:-q8_0}
IMPL=vendor/whisper.cpp/ggml/src/ggml-metal/ggml-metal-impl.h
cp "$IMPL" /tmp/ggml-metal-impl.h.orig

set_cfg() {
    sed -i '' -E \
        -e "s/^#define N_MM_NK [0-9]+/#define N_MM_NK $1/" \
        -e "s/^#define N_MM_BLOCK_X [0-9]+/#define N_MM_BLOCK_X $2/" \
        -e "s/^#define N_MM_BLOCK_Y [0-9]+/#define N_MM_BLOCK_Y $3/" \
        -e "s/^#define N_MM_SIMD_GROUP_X [0-9]+/#define N_MM_SIMD_GROUP_X $4/" \
        -e "s/^#define N_MM_SIMD_GROUP_Y [0-9]+/#define N_MM_SIMD_GROUP_Y $5/" "$IMPL"
}

for cfg in "2 4 2 2 2" "4 4 2 2 2" "8 4 2 2 2" "4 4 4 2 2" "4 2 2 2 2" "4 2 4 2 2" "8 2 2 2 2"; do
    set -- $cfg
    set_cfg "$@"
    cmake --build build/whisper -j 10 --target ggml-metal >/dev/null 2>&1 && bench/gemm/build.sh >/dev/null 2>&1 || { echo "build failed: $cfg"; continue; }
    echo "### NK=$1 BX=$2 BY=$3 SGX=$4 SGY=$5  (tile $((16*$3*$5))x$((16*$2*$4)), K $((16*$1)))"
    bench/gemm/gemm_bench "$TYPE" 8 2>/dev/null | awk 'NR>2 && ($4==128 || $4==256 || $4==1500) {printf "  %-9s N=%-5s %6.2f TFLOPS\n", $1, $4, $6}'
done

cp /tmp/ggml-metal-impl.h.orig "$IMPL"
cmake --build build/whisper -j 10 --target ggml-metal >/dev/null 2>&1 && bench/gemm/build.sh >/dev/null 2>&1
echo "restored baseline"
