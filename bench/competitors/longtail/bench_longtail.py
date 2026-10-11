#!/usr/bin/env python3
"""Long-tail Whisper large-v3-turbo runtimes on Apple Silicon.

  hf-mps             Hugging Face transformers on PyTorch MPS, fp16, sdpa attention
                     (the stack "insanely-fast-whisper" uses).
  fasterwhisper-cpu  faster-whisper / CTranslate2 on CPU (no Metal backend; int8 GEMMs
                     via Ruy, float32 GEMMs via Accelerate).

Same protocol as the other harnesses: load the model once, decode all audio to
16 kHz float32 before timing, warm up (each latency clip once), and time only
the transcription call (feature extraction + decode + detokenize). Decoding is
language "en", greedy, with no temperature fallback, no timestamps, no word
timestamps and no prompt.

Each backend runs under its own venv, so all framework imports are lazy.
"""

import argparse
import copy
import gc
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[3]  # <repo>/bench/competitors/longtail/bench_longtail.py
BENCH = REPO / "bench"
CLIPS = ["short", "medium", "long", "xlong"]
SR = 16000
WINDOW = 30 * SR  # Whisper's fixed 30 s input window


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        sys.exit(f"{path}: expected {SR} Hz, got {sr} Hz")
    return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)


def eval_audio_path(path):
    # Audio path from column 0 of the eval TSV. A relative path is taken from the repo
    # root (macrophone-bench, run from there, reads the list the same way). An absolute path
    # written by a checkout elsewhere (.../bench/data/...) is re-rooted under this one.
    p = Path(path)
    if not p.is_absolute():
        return str(REPO / p)
    if not p.exists():
        parts = p.parts
        for i in range(len(parts) - 2, 0, -1):
            if parts[i] == "bench" and parts[i + 1] == "data":
                return str(REPO.joinpath(*parts[i:]))
    return path


def clean(text):
    return " ".join(text.split())  # one line, no tabs: safe for TSV


def windows(audio):
    # Clips over 30 s are split into consecutive 30 s windows, decoded
    # independently without timestamps (no conditioning on previous text).
    return [audio[i:i + WINDOW] for i in range(0, len(audio), WINDOW)]


# ---------------------------------------------------------------- runtimes

def make_hf_mps(args):
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repo = "openai/whisper-large-v3-turbo"
    rev = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"  # the snapshot the published numbers used
    dev = torch.device("mps")
    processor = WhisperProcessor.from_pretrained(repo, revision=rev)
    hybrid = args.attn == "eager-encoder"
    model = WhisperForConditionalGeneration.from_pretrained(
        repo, revision=rev, dtype=torch.float16, attn_implementation="sdpa" if hybrid else args.attn
    ).to(dev).eval()
    if hybrid:
        # torch 2.11 MPS sdpa is ~2.5x slower than eager on the 1500-token encoder
        # self-attention, but faster in the KV-cached decoder. Attention modules read
        # self.config._attn_implementation per call, so give only the encoder an eager copy.
        enc_cfg = copy.copy(model.config)
        enc_cfg._attn_implementation = "eager"
        for layer in model.model.encoder.layers:
            layer.self_attn.config = enc_cfg
        assert model.model.encoder.layers[0].self_attn.config._attn_implementation == "eager"
        assert model.model.decoder.layers[0].self_attn.config._attn_implementation == "sdpa"
        assert model.model.decoder.layers[0].encoder_attn.config._attn_implementation == "sdpa"
    fe, tok = processor.feature_extractor, processor.tokenizer
    # temperature=None -> one greedy pass; no fallback thresholds are set.
    gen = dict(language="en", task="transcribe", return_timestamps=False,
               num_beams=1, do_sample=False)

    @torch.inference_mode()
    def decode_window(chunk):
        feats = fe(chunk, sampling_rate=SR, return_tensors="pt").input_features
        ids = model.generate(input_features=feats.to(dev, torch.float16), **gen)
        return tok.decode(ids[0].cpu(), skip_special_tokens=True)  # .cpu() syncs MPS

    def transcribe(audio):
        # transformers refuses >3000 mel frames without timestamps, so window manually.
        return " ".join(decode_window(w).strip() for w in windows(audio))

    attn = "encoder eager + decoder sdpa" if hybrid else model.config._attn_implementation
    desc = (f"transformers {transformers.__version__}, torch {torch.__version__}, "
            f"{repo}, MPS fp16, attn={attn}")
    return transcribe, desc


def make_fasterwhisper_cpu(args):
    import ctranslate2
    import faster_whisper
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    repo = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"  # faster-whisper's own "large-v3-turbo"
    path = snapshot_download(repo, revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", allow_patterns=[
        "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"])
    model = WhisperModel(path, device="cpu", compute_type=args.compute_type,
                         cpu_threads=args.threads, num_workers=1)
    opts = dict(
        language="en", task="transcribe",
        beam_size=1, best_of=1, temperature=0.0,  # single greedy pass
        compression_ratio_threshold=None, log_prob_threshold=None,
        no_speech_threshold=None,  # no fallback / no-speech skipping
        condition_on_previous_text=False, initial_prompt=None, prefix=None,
        without_timestamps=True, word_timestamps=False, vad_filter=False,
    )

    def transcribe(audio):
        segments, _ = model.transcribe(audio, **opts)
        return " ".join(s.text.strip() for s in segments)  # consuming the generator runs the decode

    desc = (f"faster-whisper {faster_whisper.__version__}, ctranslate2 {ctranslate2.__version__}, "
            f"{repo}, cpu compute_type={args.compute_type} cpu_threads={args.threads}")
    return transcribe, desc


RUNTIMES = {"hf-mps": make_hf_mps, "fasterwhisper-cpu": make_fasterwhisper_cpu}


# ---------------------------------------------------------------- protocol

def timed(transcribe, audio):
    gc.collect()
    gc.disable()
    try:
        t0 = time.perf_counter()
        text = transcribe(audio)
        ms = (time.perf_counter() - t0) * 1000.0
    finally:
        gc.enable()
    return ms, clean(text)


def write_tsv(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("\t".join(str(c) for c in row) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runtime", choices=sorted(RUNTIMES))
    ap.add_argument("--latency-out", help="write latency TSV here (skip latency if omitted)")
    ap.add_argument("--eval-out", help="write eval TSV here (skip eval if omitted)")
    ap.add_argument("--clips", default=",".join(CLIPS), help="latency clips (default: all four)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per latency clip")
    ap.add_argument("--warmup", default=",".join(CLIPS),
                    help="clips transcribed once each before timing (at least 2; default: all four, "
                         "like the mlx and whisper.cpp harnesses)")
    ap.add_argument("--eval-tsv", default=f"{BENCH}/data/eval-160.tsv")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "eager-encoder"],
                    help="hf-mps: attention implementation (eager-encoder = eager encoder + sdpa decoder)")
    ap.add_argument("--compute-type", default="int8", help="fasterwhisper-cpu: CTranslate2 compute type")
    ap.add_argument("--threads", type=int, default=4, help="fasterwhisper-cpu: cpu_threads")
    args = ap.parse_args()

    log = lambda *a: print(*a, file=sys.stderr, flush=True)

    # Decode every input before loading or timing anything.
    clips = [c for c in args.clips.split(",") if c]
    warmups = [c for c in args.warmup.split(",") if c]
    if len(warmups) < 2:
        ap.error("--warmup needs at least 2 clips")
    clip_audio = {c: load_audio(f"{BENCH}/audio/{c}.wav") for c in dict.fromkeys(warmups + clips)}
    evals = []
    if args.eval_out:
        with open(args.eval_tsv, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    path = line.rstrip("\n").split("\t")[0]
                    evals.append((os.path.basename(path), load_audio(eval_audio_path(path))))

    t0 = time.perf_counter()
    transcribe, desc = RUNTIMES[args.runtime](args)
    log(f"[{args.runtime}] {desc}")
    log(f"[{args.runtime}] loaded in {time.perf_counter() - t0:.1f} s")

    for clip in warmups:
        ms, text = timed(transcribe, clip_audio[clip])
        log(f"[{args.runtime}] warmup {clip}: {ms:.1f} ms  {text}")

    if args.latency_out:
        rows = []
        for clip in clips:
            ms_list = []
            for run in range(1, args.runs + 1):
                ms, text = timed(transcribe, clip_audio[clip])
                rows.append((clip, run, f"{ms:.2f}", text))
                ms_list.append(ms)
                log(f"[{args.runtime}] {clip} run {run}: {ms:.1f} ms  {text}")
            log(f"[{args.runtime}] {clip}: median {statistics.median(ms_list):.1f} ms")
        write_tsv(args.latency_out, rows)
        log(f"[{args.runtime}] wrote {args.latency_out}")

    if args.eval_out:
        rows = []
        for n, (name, audio) in enumerate(evals, 1):
            ms, text = timed(transcribe, audio)
            rows.append((name, f"{ms:.2f}", text))
            if n % 20 == 0 or n == len(evals):
                log(f"[{args.runtime}] eval {n}/{len(evals)}")
        write_tsv(args.eval_out, rows)
        log(f"[{args.runtime}] wrote {args.eval_out}")


if __name__ == "__main__":
    main()
