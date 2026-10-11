#!/usr/bin/env bash
# Downloads Whisper large-v3-turbo and the Silero VAD model into ~/Library/Application Support/Macrophone/models
# and quantizes turbo to q4_0 (the default: same WER as q8_0/f16 on our eval, ~30% faster decoding).
#
#   scripts/fetch-models.sh           what the app needs
#   scripts/fetch-models.sh --bench   also the weights the benchmarks compare against (q8_0, q5_0 and the
#                                     Core ML encoder used by whisper.cpp's Neural Engine variant), ~2.6 GB more
#
# Downloads are pinned to Hugging Face revisions and checked against the exact sizes benchmarked.
set -euo pipefail

CDPATH= cd -- "$(dirname -- "$0")/.."
DIR="$HOME/Library/Application Support/Macrophone/models"
mkdir -p "$DIR"

WHISPER_REV=5359861c739e955e79d9a303bcbc70fb988958b1 # huggingface.co/ggerganov/whisper.cpp
VAD_REV=9ffd54a1e1ee413ddf265af9913beaf518d1639b     # huggingface.co/ggml-org/whisper-vad

fetch() {
    local url=$1 name=$2 size=$3
    if [ -s "$DIR/$name" ] && [ "$(stat -f%z "$DIR/$name")" = "$size" ]; then
        echo "✓ $name"
        return
    fi
    echo "▸ $name"
    curl -fL --progress-bar -C - -o "$DIR/$name.part" "$url"
    if [ "$(stat -f%z "$DIR/$name.part")" != "$size" ]; then
        echo "error: $name has an unexpected size (want $size bytes)" >&2
        exit 1
    fi
    mv "$DIR/$name.part" "$DIR/$name"
}

W=https://huggingface.co/ggerganov/whisper.cpp/resolve/$WHISPER_REV
fetch "$W/ggml-large-v3-turbo.bin" ggml-large-v3-turbo.bin 1624555275
fetch "https://huggingface.co/ggml-org/whisper-vad/resolve/$VAD_REV/ggml-silero-v6.2.0.bin" ggml-silero-v6.2.0.bin 885098

if [ ! -s "$DIR/ggml-large-v3-turbo-q4_0.bin" ]; then
    [ -x build/whisper/bin/whisper-quantize ] || scripts/build-whisper.sh
    echo "▸ quantizing to q4_0"
    build/whisper/bin/whisper-quantize "$DIR/ggml-large-v3-turbo.bin" "$DIR/ggml-large-v3-turbo-q4_0.bin" q4_0 >/dev/null
fi
echo "✓ ggml-large-v3-turbo-q4_0.bin"

if [ "${1:-}" = "--bench" ]; then
    fetch "$W/ggml-large-v3-turbo-q8_0.bin" ggml-large-v3-turbo-q8_0.bin 874188075
    fetch "$W/ggml-large-v3-turbo-q5_0.bin" ggml-large-v3-turbo-q5_0.bin 574041195
    fetch "$W/ggml-large-v3-turbo-encoder.mlmodelc.zip" ggml-large-v3-turbo-encoder.mlmodelc.zip 1173393014
fi
