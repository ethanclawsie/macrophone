#!/usr/bin/env bash
# Generates the four latency-benchmark clips (16 kHz mono 16-bit WAV) with macOS text-to-speech.
# The published numbers used the system default voice on macOS 26.3; another voice or OS version gives
# slightly different audio (and timings).
set -euo pipefail
CDPATH= cd -- "$(dirname -- "$0")"
mkdir -p audio

clip() {
    say -o "audio/$1.wav" --file-format=WAVE --data-format=LEI16@16000 "$2"
    printf "%-7s %5.2f s\n" "$1" "$(afinfo "audio/$1.wav" | awk '/estimated duration/{print $3}')"
}

clip short  "Run the tests again."
clip medium "Can you refactor the use effect hook in app dot tsx so it uses a reducer instead?"
clip long   "Okay, so the bug is in the parser. When the input has a trailing comma, the tokenizer emits an empty token and then the AST builder crashes. Add a test case for that, fix it, and then run pytest with the verbose flag."
clip xlong  "Let's step back and think about the architecture. I want the audio capture to run on a real time thread that writes into a lock free ring buffer, then a worker thread pulls samples, resamples them to sixteen kilohertz, runs voice activity detection, and hands finished segments to the whisper engine. Keep the model loaded and warm the GPU when the hotkey goes down, and measure every stage with signposts so we can see where the latency goes."
