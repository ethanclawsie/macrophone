#!/usr/bin/env bash
# Downloads LibriSpeech test-clean (346 MB, CC BY 4.0) and builds the 160-utterance accuracy set:
# 40 utterances from each length bucket (<3 s, 3-6 s, 6-12 s, 12 s+), evenly spaced in the sorted manifest,
# so every machine gets the same list (bench/eval-160-ids.txt).
set -euo pipefail
CDPATH= cd -- "$(dirname -- "$0")/.."

mkdir -p bench/data/librispeech
if [ ! -d bench/data/librispeech/LibriSpeech/test-clean ]; then
    curl -fL --progress-bar -o bench/data/librispeech/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
    tar xzf bench/data/librispeech/test-clean.tar.gz -C bench/data/librispeech
    rm bench/data/librispeech/test-clean.tar.gz
fi

swift build -c release --product macrophone-bench >/dev/null
.build/release/macrophone-bench manifest bench/data/librispeech/LibriSpeech/test-clean bench/data/manifest.tsv
.build/release/macrophone-bench pick --per-bucket 40 bench/data/eval-160.tsv

# The selection must match the published one.
if diff -q <(cut -f1 bench/data/eval-160.tsv | xargs -n1 basename) bench/eval-160-ids.txt >/dev/null; then
    echo "✓ bench/data/eval-160.tsv matches bench/eval-160-ids.txt"
else
    echo "warning: the selected utterances differ from bench/eval-160-ids.txt" >&2
fi
