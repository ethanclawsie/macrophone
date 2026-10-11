#!/usr/bin/env bash
# Builds whisper.cpp (vendor/whisper.cpp, pinned + patched) as static libraries into build/whisper.
#
# The Metal shaders are embedded and compiled at runtime, so this needs only the Command Line
# Tools (no offline `metal` compiler). On M5 that runtime path also enables ggml's Metal 4 tensor
# kernels, which run matmuls on the GPU Neural Accelerators (~2x faster Whisper encoder).
set -euo pipefail

CDPATH= cd -- "$(dirname -- "$0")/.."
WHISPER_COMMIT=d1be6fde11ac6e0407606b4e42fe72d34add8037 # v1.9.5

if [ ! -d vendor/whisper.cpp ]; then
    git init -q vendor/whisper.cpp
    git -C vendor/whisper.cpp fetch -q --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_COMMIT"
    git -C vendor/whisper.cpp -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
if [ "$(git -C vendor/whisper.cpp rev-parse HEAD)" != "$WHISPER_COMMIT" ]; then
    echo "error: vendor/whisper.cpp is not at v1.9.5 ($WHISPER_COMMIT); delete vendor/ and re-run" >&2
    exit 1
fi

for patch in patches/*.patch; do
    [ -e "$patch" ] || continue
    if git -C vendor/whisper.cpp apply --reverse --check "../../$patch" 2>/dev/null; then
        continue # already applied
    fi
    if ! git -C vendor/whisper.cpp apply "../../$patch"; then
        echo "error: $patch doesn't apply to vendor/whisper.cpp (after an update: rm -rf vendor build, then re-run)" >&2
        exit 1
    fi
    echo "▸ applied $patch"
done

GENERATOR="Unix Makefiles"
command -v ninja >/dev/null && GENERATOR=Ninja

# -Wno-deprecated: upstream's cmake_minimum_required warning. OpenMP isn't available with Apple clang anyway.
cmake -S vendor/whisper.cpp -B build/whisper -G "$GENERATOR" -Wno-deprecated \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENMP=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_NATIVE=ON \
    -DWHISPER_BUILD_EXAMPLES=ON \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_SERVER=OFF \
    -DWHISPER_SDL2=OFF >/dev/null

cmake --build build/whisper -j "$(sysctl -n hw.ncpu)" --target whisper whisper-cli whisper-bench whisper-quantize

# SwiftPM doesn't track the prebuilt archives: make it relink.
touch Sources/CWhisper/shim.c
echo "✓ build/whisper"
