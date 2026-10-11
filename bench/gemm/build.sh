#!/usr/bin/env bash
# Builds the Metal kernel micro-benchmarks (GEMM, mat-vec, one encoder layer, op chains) against the
# static ggml libraries in build/whisper.
set -euo pipefail
CDPATH= cd -- "$(dirname -- "$0")/../.."
B=build/whisper/ggml/src
for b in gemm_bench gemv_bench layer_bench chain_bench; do
    clang++ -std=c++17 -O2 "bench/gemm/$b.cpp" -o "bench/gemm/$b" \
        -Ivendor/whisper.cpp/ggml/include \
        -L$B -L$B/ggml-metal -L$B/ggml-blas \
        -lggml -lggml-base -lggml-cpu -lggml-metal -lggml-blas \
        -framework Metal -framework Foundation -framework Accelerate
    echo "✓ bench/gemm/$b"
done
