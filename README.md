# Macrophone

Hold-to-talk dictation for macOS, built for talking to coding agents. Hold **Right ⌥**, speak, let go: the
text is pasted wherever your cursor is — a terminal running Claude Code, VS Code, a browser. Hold **⇧** as you
let go to also press Return and send it.

It runs **Whisper large-v3-turbo** (4-bit q4_0 weights) locally on Apple Silicon and is built around one
number: the time between letting go of the key and the text appearing. On a MacBook Pro M5 it transcribes a
finished recording faster than every other large-v3-turbo runtime we benchmarked (whisper.cpp, MLX,
WhisperKit, transformers and faster-whisper; 14 configurations in all), at the same accuracy:

| runtime (its fastest configuration) | 1.1 s clip | 4.4 s | 13.1 s | 24.6 s | WER (≤ 30 s) |
|---|---:|---:|---:|---:|---:|
| **Macrophone** (q4_0) | **45 ms** | **87 ms** | **240 ms** | **401 ms** | 1.72% |
| MLX Whisper (4-bit) | 304 ms | 327 ms | 381 ms | 425 ms | 1.96% |
| whisper.cpp (Metal, q8_0) | 360 ms | 395 ms | 478 ms | 571 ms | 1.66% |
| Hugging Face transformers (MPS, fp16) | 368 ms | 418 ms | 536 ms | 654 ms | 1.69% |
| WhisperKit (Neural Engine encoder, GPU decoder) | 383 ms | 495 ms | 753 ms | 1008 ms | 1.81% |
| faster-whisper (CPU) | 2111 ms | 2208 ms | 2605 ms | 2893 ms | 1.72% |

*Time to transcribe a finished recording, model loaded and warm (median of 5 runs). WER = word error rate on
the 158 LibriSpeech utterances up to 30 s long, lower is better; the differences are a few words out of 3,314
and within noise. Each row is the runtime's fastest configuration by median latency on that evaluation set.
All 14 configurations and the method are under [Benchmarks](#benchmarks).*

The lead is largest for short dictation (6.7× on the 1.1 s command) and shrinks with length, down to 6% on
the 24.6 s clip. In the app it is faster still: speech is transcribed while you are still holding the key, so
when you let go after a natural pause the text is usually already there.

Requires an Apple Silicon Mac on macOS 26 (fastest on M5-generation chips) and a build from source; there is
no prebuilt download. It's a native Swift menu bar app: no Dock icon, no network access, no CPU use while idle
(after a 3-minute GPU keep-alive). It builds without Xcode.app — the Command Line Tools plus CMake and Ninja
are enough.

## Install

Requirements:
- An Apple Silicon Mac on macOS 26. The engine is tuned for M5-generation GPUs (their Neural Accelerators);
  M1–M4 should work but are untested and won't reach these numbers.
- The Xcode Command Line Tools (`xcode-select --install`) and CMake + Ninja (`brew install cmake ninja`).
- About 2.1 GB of disk for the models (a 1.6 GB download), kept in
  `~/Library/Application Support/Macrophone/models`.

```sh
git clone https://github.com/ethanclawsie/macrophone && cd macrophone
scripts/build-whisper.sh   # fetches whisper.cpp v1.9.5, applies patches/, builds static libraries
scripts/fetch-models.sh    # Whisper large-v3-turbo (→ q4_0) + Silero VAD into ~/Library/Application Support/Macrophone/models
scripts/build-app.sh       # → dist/Macrophone.app (ad-hoc signed)
open dist/Macrophone.app
```

A waveform icon appears in the menu bar (an hourglass while the model loads), and macOS asks for three
permissions: **Microphone** (to hear you), **Input Monitoring** (to see the hotkey) and **Accessibility** (to
paste). The menu lists anything still missing. If the hotkey doesn't respond after granting Input Monitoring,
quit and reopen the app.

Build it on the Mac that will run it: the engine is compiled for the build machine's CPU. The app is ad-hoc
signed with a designated requirement that pins only its bundle identifier, so rebuilding keeps the
permissions; the flip side is that any ad-hoc-signed app with the identifier `com.ethanclawsie.macrophone`
would inherit them.

To update: `git pull && rm -rf vendor build && scripts/build-whisper.sh && scripts/build-app.sh`.

## Use

| action | result |
|---|---|
| hold Right ⌥, speak, release | the text is pasted at the cursor |
| …release while holding ⇧ | pasted, then Return |
| press any other key while holding ⌥ (e.g. ⌥← word jump) | discarded — you were using ⌥ for a shortcut |
| tap ⌥ for less than ~0.2 s | ignored |

- Pause a beat before releasing and the text appears instantly (it was transcribed during the pause).
- Your clipboard is preserved: the text is offered lazily, Macrophone sees when the target app reads it, and
  puts your previous clipboard back 200 ms later. Entries are marked transient so clipboard managers skip them.
- If a Bluetooth headset is the system input, Macrophone records from the built-in mic instead (opening a
  headset mic switches it to the slow, low-quality call profile).
- Each dictation's latency is shown in the menu and logged to `~/Library/Logs/Macrophone/latency.tsv`.

Settings live in `~/Library/Application Support/Macrophone/config.json`, which Macrophone writes on first
launch with every setting at its default (menu → Open Config File, then Reload Config). Keys you delete fall
back to their defaults. For example:

```json
{
  "hotkey": "rightOption",
  "vocabulary": ["Claude Code", "PyTest", "useEffect", "kubectl"],
  "replacements": { "clod code": "Claude Code" },
  "micKeepAliveSeconds": 0
}
```

- `hotkey` is `rightOption`, `rightCommand`, `rightControl` or `fn` (for `fn`, set System Settings → Keyboard →
  "Press 🌐 key to" → Do Nothing).
- `vocabulary` is given to Whisper as prompt text and biases how it spells names and jargon.
- `replacements` are case-insensitive phrase replacements applied to the transcript.
- `micKeepAliveSeconds` keeps the microphone running that long after each dictation, so the next one doesn't
  wait for it to start (macOS shows its microphone indicator meanwhile).

`modelPath`, `vadModelPath`, `language`, `threads` and `preferBuiltInMic` take effect after a restart;
everything else applies on Reload Config. If the file doesn't parse, the defaults are used and the file is
left as it is.

Limitations: the defaults are tuned for English; paste sends ⌘ with the key code of V on a QWERTY keyboard,
so on layouts that move V (Dvorak, for example) it may not paste; it has been tested on one machine, a
MacBook Pro M5 on macOS 26.3.

## Benchmarks

### What was compared

Every configuration runs the same model, **OpenAI Whisper large-v3-turbo**. Each runtime is tested in its
default setup plus the variants that make it fastest for dictation (quantized weights, GPU vs Neural Engine).
Nothing that costs accuracy is used, such as shrinking whisper.cpp's `audio_ctx` (see
[How it's fast](#how-its-fast)). 14 configurations of 6 runtimes, 2 of them Macrophone's own:

| name | runtime | weights | runs on | notes |
|---|---|---|---|---|
| `macrophone` | this repo (whisper.cpp v1.9.5 + [patches](patches/)) | q4_0 | GPU (Metal, M5 Neural Accelerators) | silence-tail cache, see [How it's fast](#how-its-fast) |
| `macrophone-q8` | same | q8_0 | GPU | |
| `whispercpp-metal-f16` / `-q8_0` / `-q5_0` | stock [whisper.cpp](https://github.com/ggml-org/whisper.cpp) v1.9.5 | f16 / q8_0 / q5_0 | GPU (Metal, Neural Accelerators on) | flash attention on |
| `whispercpp-coreml-f16` | stock whisper.cpp v1.9.5, `WHISPER_COREML=ON` | f16 + Core ML encoder | encoder through Core ML (all compute units), decoder on GPU | |
| `whispercpp-metal-f16-cli-defaults` | stock whisper.cpp v1.9.5 | f16 | GPU | whisper-cli's default decoding: beam search (5), temperature fallback, timestamps — for reference |
| `whisperkit` | [WhisperKit](https://github.com/argmaxinc/argmax-oss-swift) 1.1.1 | `openai_whisper-large-v3-v20240930_turbo` (Core ML) | encoder + decoder on the Neural Engine (default) | |
| `whisperkit-gpudec` | WhisperKit 1.1.1 | same | encoder on the Neural Engine, decoder on GPU | |
| `mlx` / `mlx-8bit` / `mlx-4bit` | [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) 0.4.3, MLX 0.32.3 | `mlx-community/whisper-large-v3-turbo` fp16 / 8-bit / 4-bit | GPU (Neural Accelerators on) | |
| `hf-mps` | Hugging Face transformers 5.19, PyTorch 2.11 | `openai/whisper-large-v3-turbo` fp16 | GPU (MPS) | eager attention in the encoder (faster than sdpa on MPS); audio over 30 s is split into 30 s windows by the harness |
| `fasterwhisper-cpu` | faster-whisper 1.2.1 (CTranslate2 4.8.2) | CTranslate2 conversion, computed in fp32 | CPU (`cpu_threads` 4; matrix products via Accelerate) | CTranslate2 has no Metal backend |

q8_0 / q5_0 / q4_0 are ggml's 8-, 5- and 4-bit block-quantized weights; f16 / fp16 is unquantized half
precision. The Neural Engine is Apple's separate ML accelerator, reached through Core ML; the Neural
Accelerators are the matrix units inside each M5 GPU core. MPS is PyTorch's Metal backend.

Decoding: language forced to English (no detection), greedy (temperature 0, no temperature fallback), no
prompt or previous-text conditioning, no timestamp tokens — except `whispercpp-metal-f16-cli-defaults`, which
keeps whisper-cli's defaults, and Macrophone, which differs in four ways: it decodes with timestamp tokens (as
one segment up to 30 s, with whisper.cpp's seek-to-the-last-timestamp beyond that), suppresses non-speech
tokens (whisper.cpp's `suppress_nst`), caps the output for clips up to 30 s at 24 + 10 tokens per second of
audio, and normalizes the log-mel spectrogram to a fixed peak (its silence cache depends on it). The exact
per-runtime options are in each harness under [`bench/competitors/`](bench/competitors/).

### How we benchmarked

- **Machine:** MacBook Pro, Apple M5 (4 performance + 6 efficiency CPU cores, 10-core GPU with Neural
  Accelerators, 16-core Neural Engine), 24 GB, macOS 26.3. Normal desktop use; the same GPU drives a 4K
  external display, so expect a few percent of run-to-run noise.
- **What is timed:** only the transcription call — in-memory 16 kHz float32 audio in, final text out — with the
  model loaded and warmed up by untimed transcriptions. The other harnesses transcribe every latency clip once
  before timing anything (WhisperKit, whose Core ML models have a fixed 30 s input shape, warms up twice on one
  clip); Macrophone's warms each clip right before its runs, after two start-up passes. Model loading, file
  decoding and process start-up are excluded everywhere.
- **Latency:** four clips of synthetic speech that sound like prompts to a coding agent — 1.1, 4.4, 13.1 and
  24.6 s, generated with macOS text-to-speech in the default voice on macOS 26.3
  ([`bench/make-audio.sh`](bench/make-audio.sh)). Each is transcribed 5 times; the tables show the median.
- **Accuracy:** 160 utterances of real read speech from
  [LibriSpeech test-clean](https://www.openslr.org/12) — 40 from each length bucket (<3 s, 3–6 s, 6–12 s,
  12 s+), taking every k-th utterance of the path-sorted bucket (k = bucket size ÷ 40, rounded down) so the
  selection is deterministic. The list is committed as [`bench/eval-160-ids.txt`](bench/eval-160-ids.txt);
  [`bench/fetch-data.sh`](bench/fetch-data.sh) rebuilds it and checks it against that file. Each utterance is
  transcribed once. Two of them are longer than 30 s (one Whisper window), so WER is reported over all 160 and
  over the 158 up to 30 s. We also report the median latency over the 160.
- **Scoring:** word error rate = (substituted + deleted + inserted words) / reference words, after both texts
  go through Whisper's English text normalizer (the `whisper-normalizer` 0.1.15 package; the Open ASR
  Leaderboard uses the same normalization), so formatting ("2000" vs "two thousand", "luster" vs "lustre",
  casing, punctuation) doesn't count as an error ([`bench/score.py`](bench/score.py)).
- **Isolation:** configurations run one at a time, never concurrently. [`bench/run-all.sh`](bench/run-all.sh)
  waits 20 s between harnesses, and the whisper.cpp harness also waits between its variants; the WhisperKit,
  MLX and transformers / faster-whisper variants run back to back.
- **Provenance:** the competitor rows come from one run of `bench/run-all.sh`. Macrophone's two rows were
  measured with the same protocol about 45 minutes later, after the final engine build. Since then the
  harnesses have only been changed to find their files relative to the repo.
- Every harness writes the same two files — `bench/results/<name>-latency.tsv` (clip, run, ms, text) and
  `<name>-eval.tsv` (utterance, ms, text) — and the tables below are computed from those files, which are
  committed in [`bench/results/`](bench/results/).

### Results

Latency (ms, median of 5 runs, model warm):

| configuration | 1.1 s | 4.4 s | 13.1 s | 24.6 s |
|---|---:|---:|---:|---:|
| **macrophone** | **45** | **87** | **240** | **401** |
| macrophone-q8 | 50 | 102 | 283 | 473 |
| mlx | 299 | 349 | 472 | 583 |
| mlx-4bit | 304 | 327 | 381 | 425 |
| mlx-8bit | 319 | 351 | 429 | 498 |
| whispercpp-metal-q8_0 | 360 | 395 | 478 | 571 |
| whispercpp-metal-f16 | 366 | 417 | 541 | 664 |
| hf-mps | 368 | 418 | 536 | 654 |
| whispercpp-metal-q5_0 | 373 | 405 | 475 | 550 |
| whispercpp-metal-f16-cli-defaults | 378 | 456 | 614 | 788 |
| whisperkit-gpudec | 383 | 495 | 753 | 1008 |
| whisperkit | 394 | 538 | 866 | 1187 |
| whispercpp-coreml-f16 | 442 | 498 | 637 | 756 |
| fasterwhisper-cpu | 2111 | 2208 | 2605 | 2893 |

Accuracy on LibriSpeech test-clean (WER %, lower is better), and median latency over the 160 utterances:

| configuration | WER ≤ 30 s (158) | WER, all 160 | <3 s | 3–6 s | 6–12 s | 12 s+ | median latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| whispercpp-metal-q8_0 | 1.66 | 2.15 | 1.47 | 3.34 | 1.90 | 2.06 | 399 ms |
| whispercpp-metal-f16 | 1.66 | 2.15 | 1.47 | 3.34 | 1.90 | 2.06 | 420 ms |
| mlx | 1.69 | 2.18 | 1.47 | 3.13 | 2.01 | 2.12 | 350 ms |
| hf-mps | 1.69 | 2.18 | 1.47 | 3.34 | 2.01 | 2.06 | 420 ms |
| **macrophone** | 1.72 | 1.74 | 1.47 | 2.92 | 1.90 | 1.37 | **111 ms** |
| mlx-8bit | 1.72 | 2.21 | 1.47 | 3.13 | 2.12 | 2.12 | 349 ms |
| whispercpp-metal-q5_0 | 1.72 | 2.21 | 1.47 | 3.34 | 2.01 | 2.12 | 406 ms |
| fasterwhisper-cpu | 1.72 | 2.21 | 1.47 | 3.13 | 2.12 | 2.12 | 2214 ms |
| macrophone-q8 | 1.75 | 1.77 | 1.84 | 2.92 | 2.12 | 1.26 | 130 ms |
| whispercpp-coreml-f16 | 1.78 | 2.26 | 1.47 | 3.34 | 2.01 | 2.23 | 503 ms |
| whisperkit | 1.78 | 2.29 | 1.47 | 3.34 | 2.12 | 2.23 | 537 ms |
| whisperkit-gpudec | 1.81 | 2.32 | 1.47 | 3.34 | 2.12 | 2.29 | 489 ms |
| whispercpp-metal-f16-cli-defaults | 1.90 | 1.92 | 1.47 | 3.34 | 2.22 | 1.43 | 448 ms |
| mlx-4bit | 1.96 | 2.44 | 1.47 | 3.13 | 2.33 | 2.46 | 322 ms |

Macrophone transcribes the median eval utterance 2.9× faster than the fastest other runtime (111 vs 322 ms,
MLX 4-bit) and the 1.1 s command 6.7× faster (45 vs 299 ms, MLX fp16). The lead shrinks with length: 3.7× on
the 4.4 s clip, 1.6× on 13.1 s and 6% on 24.6 s (401 vs 425 ms, MLX 4-bit). With the same q8_0 weights as
stock whisper.cpp, `macrophone-q8` is 7.1× faster on the 1.1 s clip (50 vs 360 ms) and 1.2× on the 24.6 s
one (473 vs 571 ms).

Accuracy is the same within noise: on the 158 utterances up to 30 s, WER ranges from 1.66% to 1.96% across
all 14 configurations, and Macrophone's 1.72% is 2 word errors more than the best (57 vs 55 errors in 3,314
words).

Things to know when reading these numbers:
- They are from one machine, a base M5. The runtimes scale differently on other chips (the Neural Engine
  results, for instance, would compare differently on an M5 Pro/Max with more GPU cores).
- The two utterances longer than 30 s explain the gap between the two WER columns. Configurations that decode
  with timestamp tokens (Macrophone and whisper-cli's defaults) continue the second window from the last
  timestamp; the others move on in fixed 30 s steps and lose the words around the cut (15–16 word errors on
  the 30.8 s utterance, against none for the two timestamp configurations). That comes from how the harnesses
  were configured, not from the runtimes, and it doesn't affect the app, which never transcribes more than
  24 s at once.
- Macrophone's rows come from `macrophone-bench protocol`, which runs the app's engine
  (`MacrophoneCore`'s `WhisperEngine`) with the app's defaults (q4_0, silence-tail window, 4 threads), one
  transcription per recording: no draft tokens, no prompt, nothing carried over between runs. The app's
  streaming pipeline is measured separately below.
- A WER difference of 0.1 points is about 3 words out of the 3,314 in the 158 utterances up to 30 s.

### In the app: time from key release to text

The tables above time one transcription of a finished recording, which is all a non-streaming engine can do
on key release. The app instead transcribes at every pause of 160 ms or more while you are still talking
(Silero VAD runs on the audio as it arrives), commits text once the open segment is at least 6 s long and a
pause lasts at least 300 ms, and reuses each pass's tokens as a draft for the next one. On release it either
already has the text, or transcribes only the speech after the last pass (when that pass ended at a pause of
300 ms or more), or otherwise the open segment since the last commit.

Measured with `macrophone-bench simulate bench/audio/*.wav` (add `--tail-ms 300` for the right-hand column),
which feeds the audio in real time through the same pipeline. Median (min–max) over 8 and 7 runs from two
sessions; the raw output is in
[`bench/results/macrophone-simulate.txt`](bench/results/macrophone-simulate.txt):

| clip | release right as you stop talking | release 300 ms after the last word |
|---|---:|---:|
| 1.1 s | 64 ms (60–100) | 0 ms (every run) |
| 4.4 s (no pauses) | 111 ms (105–155) | 14 ms (13–58) |
| 13.1 s | 94 ms (88–166), last 2.4 s only | 0 ms (0–23; 0 in 5 of 7 runs) |
| 24.6 s | 111 ms (105–152), last 4.5 s only | 0 ms (every run) |

Each pass here is slower than in the warm benchmark (64 vs 45 ms for the 1.1 s clip) because the GPU and CPU
clock down between passes while you talk; spinning on GPU completion instead of sleeping (see below) removed
much of that penalty. The 13.1 s and 24.6 s clips still finish sooner than in the benchmark because only the
speech after the last pause is left to transcribe.

### Reproducing

Needs, besides the Install requirements: [uv](https://docs.astral.sh/uv/) (for the Python harnesses and the
WER scorer, so also for `bench/run-all.sh macrophone`), `git`, `unzip`, network access, and ~15 GB of disk for
the other runtimes' models and environments. Close other GPU-heavy apps while it runs.

```sh
scripts/build-whisper.sh
scripts/fetch-models.sh --bench   # + q8_0, q5_0 and the Core ML encoder (pinned revisions, size-checked)
bench/make-audio.sh               # latency clips → bench/audio/ (prints their lengths; another voice gives different clips)
bench/fetch-data.sh               # LibriSpeech test-clean → bench/data/eval-160.tsv (checked against bench/eval-160-ids.txt)

# one-time setup of the other runtimes
bench/competitors/whispercpp/setup.sh --ane-compile   # stock whisper.cpp v1.9.5, Metal + Core ML builds
(cd bench/competitors/whisperkit && swift build -c release && \
  .build/release/whisperkit-bench download --hf-base "$PWD/hf" --variant openai_whisper-large-v3-v20240930_turbo)
bench/competitors/mlx/setup.sh                        # uv venv + pinned model snapshots
bench/competitors/longtail/setup.sh                   # pinned venvs; pinned models download on the first (untimed) load

bench/run-all.sh          # everything, one at a time (~30 min); `bench/run-all.sh macrophone` for Macrophone only
```

`run-all.sh` writes `bench/results/wer.txt` and `bench/results/latency-table.md`. The first load of each
runtime compiles its GPU / Neural Engine kernels (up to ~90 s for WhisperKit's encoder); that is not timed.

`run-all.sh` doesn't stop when a harness fails: it logs `!!! failed` to `bench/results/run-all.log` and moves
on, and the scoring then includes that harness's committed result files as if they were yours.
`git status bench/results` shows which files your run replaced. Likewise, `bench/run-all.sh macrophone` ranks
your Macrophone run against the committed numbers for the other runtimes. To re-score the committed results
without running anything (after `bench/fetch-data.sh`, which provides the references):
`uv run --no-project --with whisper-normalizer==0.1.15 python -I bench/score.py`.

## How it's fast

The latency that matters is key release → text. Three levels:

**1. Don't encode the padding.** Whisper always encodes a 30 s window; for a 2 s command 93% of that work is
silence, and in stock whisper.cpp it costs ~345 ms on the M5's GPU no matter how short you speak. Simply
shrinking the window (whisper.cpp's `audio_ctx`) makes turbo loop and hallucinate (19–31% WER in our
development runs). Two reasons:
- **A whisper.cpp bug.** With flash attention, both the encoder and the decoder attend over a key/value
  buffer padded to a multiple of 256 frames, but only the frames of the current encode are written — the rest
  hold stale keys/values from the *previous* clip, unmasked. Any reduced window decoded old audio. Fixed with
  masks in [the patch](patches/).
- **The model needs to see silence.** It learned "speech has ended" from long silent tails. So Macrophone
  keeps a **silence-tail cache**: one encode of pure silence at start-up stores every encoder layer's
  keys/values and the decoder's cross-attention keys/values. Each pass computes only the frames that contain
  audio; at every layer those frames still attend over all 1500 positions, with the silent part's keys/values
  taken from the cache (an approximation: they were computed without the speech in context). The log-mel
  spectrogram is normalized to a fixed peak so silence always has the same value and one cache fits every
  clip.

  Result: ~33 ms of encoder time for a short clip, instead of ~306 ms for the full window in the same engine
  (~345 ms in stock whisper.cpp). Accuracy holds: with the same q8_0 weights, WER on the 158 utterances up to
  30 s is 1.75%, against 1.66% for stock whisper.cpp's full-window encode (58 vs 55 word errors in 3,314
  words; the decoding settings also differ, see above).

**2. Hide the rest.** Transcription happens at pauses while you're still holding the key (above). Decoding
uses timestamp mode, so whisper.cpp stops as soon as a predicted timestamp reaches the end of the audio instead
of waiting for the end-of-text token.

**3. Remove what profiling found.** The accuracy of the end result is in the tables above.

| change | effect |
|---|---|
| q4_0 weights (the decoder is memory-latency bound) | ~30% faster decoding; WER 1.77% → 1.74% (30 of 160 transcripts change, 8 after normalization) |
| Vectorized contiguous copy kernel (ggml's ran at 14–28 GB/s: 64-bit div/mod per element) | copies 4–8× faster (~95–120 GB/s) |
| Spectrogram as two SGEMMs on Accelerate (instead of a threaded scalar FFT) | 6–13 ms → 1 ms on 25 s of audio |
| Vectorized softmax/argmax, suppress-token ids looked up once | sampling 22 → 9 ms on 25 s |
| Bias/residual/GELU fused into the decoder's mat-vec and the encoder's tensor-matmul epilogues | fewer kernels and memory passes |
| F16 activations into the M5 tensor matmuls on long inputs | half the operand traffic |
| Spin on Metal command-buffer completion instead of sleeping | first pass after a short idle, 4.4 s clip: +54% → +20–26% over warm (the 1.1 s clip still pays +36–45%) |
| Memory-residency thread sleeps once the 3-minute keep-alive after the last pass expires (stock ggml wakes it every 5 ms forever) | 0% CPU when idle |

Things that looked good in micro-benchmarks but did nothing end to end — matmul tile tuning, mat-vec work
splits, a GPU heartbeat to keep clocks up — were left out.

## Development

```sh
scripts/build-whisper.sh && bench/make-audio.sh                 # once: the libraries and the latency clips
swift build -c release --product macrophone-bench
.build/release/macrophone-bench transcribe bench/audio/*.wav     # per-stage timing (mel, encode, decode, sampling)
.build/release/macrophone-bench simulate --tail-ms 300 FILE.wav  # the whole dictation pipeline, audio fed in real time
.build/release/macrophone-bench eval --per-bucket 40             # quick WER + latency (after bench/fetch-data.sh)
bench/gemm/build.sh                                              # Metal kernel micro-benchmarks
```

`macrophone-bench eval` uses a simpler text normalizer than `bench/score.py`, so its WER reads higher (2.54%
vs 1.74% for the same transcripts).

| path | contents |
|---|---|
| `Sources/Macrophone/` | menu bar app: hotkey (event tap), paste with clipboard restore, permissions, menu |
| `Sources/MacrophoneCore/` | engine wrapper, streaming VAD, dictation scheduler (speculation, commits, drafts), mic capture |
| `Sources/CMacrophoneRT/` | lock-free ring buffer written from the Core Audio I/O thread |
| `Sources/CWhisper/` | whisper.cpp headers and link settings for the static libraries |
| `Sources/macrophone-bench/` | benchmark / evaluation CLI |
| `patches/` | changes to whisper.cpp v1.9.5 and ggml-metal (applied by `scripts/build-whisper.sh`) |
| `scripts/` | build and model-download scripts |
| `Resources/` | the app's Info.plist |
| `bench/` | benchmark harnesses, scripts and results |

## Credits

[whisper.cpp / ggml](https://github.com/ggml-org/whisper.cpp) (MIT, see [patches/LICENSE-whisper.cpp](patches/LICENSE-whisper.cpp)),
[OpenAI Whisper](https://github.com/openai/whisper) large-v3-turbo weights (MIT),
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT). The
[LibriSpeech ASR corpus](https://www.openslr.org/12) by V. Panayotov, G. Chen, D. Povey and S. Khudanpur
(ICASSP 2015), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), is used for evaluation:
`bench/eval-160-ids.txt` lists 160 test-clean utterance IDs and `bench/results/*-eval.tsv` contain machine
transcriptions of them. The benchmark harnesses use
[WhisperKit](https://github.com/argmaxinc/argmax-oss-swift), [MLX](https://github.com/ml-explore/mlx),
[Hugging Face transformers](https://github.com/huggingface/transformers) and
[faster-whisper](https://github.com/SYSTRAN/faster-whisper).
