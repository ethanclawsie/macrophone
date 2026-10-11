#!/usr/bin/env python3
"""MLX Whisper (mlx-whisper on PyPI) large-v3-turbo harness for the macrophone engine comparison.

  bench.py --fetch                 download the pinned model snapshots into $HF_HOME (network)
  bench.py --variant V --latency --eval [--out-dir DIR] [--clips a,b] [--runs N] [--timestamps]

Every variant is decoded with: language="en", task="transcribe", temperature=0.0 (a single value,
so no fallback), condition_on_previous_text=False, word_timestamps=False, no initial prompt,
without_timestamps=True and fp16=True. mlx_whisper pads every window to 30 s; with
<|notimestamps|> a clip of <= 30 s is exactly one encoder pass plus one greedy decode.

Timing covers only mlx_whisper.transcribe() on an already-decoded 16 kHz mono float32 numpy
array, with the model loaded and warmed up. The returned text is a host-side str (tokens are
pulled with .tolist()), and mx.synchronize() runs inside the timed region so that the extra
decoder step mlx_whisper queues past EOT is finished before the clock stops.
"""

import argparse
import gc
import math
import os
import sys
import time
from pathlib import Path

# Everything is located relative to this file (<repo>/bench/competitors/mlx/bench.py), so the repo
# can be cloned anywhere.
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
BENCH = REPO / "bench"
AUDIO_DIR = BENCH / "audio"
EVAL_LIST = BENCH / "data" / "eval-160.tsv"
RESULTS = BENCH / "results"
LATENCY_CLIPS = ["short", "medium", "long", "xlong"]

# Output label -> (Hugging Face repo, pinned commit). All three are Whisper large-v3-turbo.
# -8bit/-4bit are mlx-community's mlx-audio-plus 0.1.4 conversions (affine, group size 64); the
# 2024 -q4 repo is structurally identical to -4bit and dequantizes to the same weights.
VARIANTS = {
    "mlx": ("mlx-community/whisper-large-v3-turbo", "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"),
    "mlx-8bit": ("mlx-community/whisper-large-v3-turbo-8bit", "62103fc276a35fdc76e318f314d8ff47987fba89"),
    "mlx-4bit": ("mlx-community/whisper-large-v3-turbo-4bit", "0f058d38170d183f9fdee07908f5b515d91793a8"),
}

SAMPLE_RATE = 16_000


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fetch", action="store_true", help="download every variant's pinned snapshot, then exit")
    p.add_argument("--variant", choices=sorted(VARIANTS))
    p.add_argument("--latency", action="store_true", help="latency clips, --runs timed runs each")
    p.add_argument("--eval", action="store_true", help="eval-160 set, one timed run per file")
    p.add_argument("--eval-list", type=Path, default=EVAL_LIST,
                   help="TSV: audio path (absolute, or relative to the repo root), seconds, reference")
    p.add_argument("--out-dir", type=Path, default=RESULTS)
    p.add_argument("--clips", default=",".join(LATENCY_CLIPS), help="latency clips (names in bench/audio)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", default=",".join(LATENCY_CLIPS),
                   help="clips transcribed once each before timing (at least 2)")
    p.add_argument("--timestamps", action="store_true",
                   help="decode with timestamp tokens instead of <|notimestamps|> (A/B only)")
    a = p.parse_args()
    if not a.fetch and (not a.variant or not (a.latency or a.eval)):
        p.error("give --fetch, or --variant with --latency and/or --eval")
    if not a.fetch and len([c for c in a.warmup.split(",") if c]) < 2:
        p.error("--warmup needs at least 2 clips")
    return a


args = parse_args()
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
if not args.fetch:
    os.environ["HF_HUB_OFFLINE"] = "1"  # read before huggingface_hub is imported

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402


def model_dir(variant: str, download: bool = False) -> Path:
    """Local directory that mlx_whisper.load_models.load_model() accepts for this variant."""
    repo, revision = VARIANTS[variant]
    snap = Path(snapshot_download(repo, revision=revision, local_files_only=not download))
    if (snap / "weights.safetensors").exists() or (snap / "weights.npz").exists():
        return snap
    # mlx-audio-plus conversions: same config.json schema and tensor names as mlx_whisper, but the
    # weights file is called model.safetensors, which load_model() doesn't look for. Alias it.
    alias = ROOT / "models" / variant
    alias.mkdir(parents=True, exist_ok=True)
    for src, dst in (("config.json", "config.json"), ("model.safetensors", "weights.safetensors")):
        target, link = (snap / src).resolve(strict=True), alias / dst
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return alias


def eval_audio_path(entry: str, eval_list: Path) -> Path:
    """Audio file named in column 1 of an eval-list row. An absolute path is used as written; a
    relative one is looked up under the repo root, then next to the eval list itself."""
    p = Path(entry)
    if p.is_absolute():
        return p
    candidates = (REPO / p, eval_list.resolve().parent / p)
    return next((c for c in candidates if c.exists()), candidates[0])


def load_16k_mono(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = math.gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g)
    return np.ascontiguousarray(data, dtype=np.float32)


def clean(text: str) -> str:
    return " ".join(text.split())  # one TSV field: no tabs/newlines, no leading space


class Writer:
    """Rows go to NAME.partial as they're produced; renamed to NAME once complete."""

    def __init__(self, path: Path):
        self.path, self.tmp = path, path.with_name(path.name + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.tmp, "w", encoding="utf-8")

    def row(self, *fields):
        self.f.write("\t".join(str(x) for x in fields) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
        os.replace(self.tmp, self.path)
        log(f"wrote {self.path}")


def main():
    if args.fetch:
        for v in VARIANTS:
            d = model_dir(v, download=True)
            log(f"{v}: {VARIANTS[v][0]}@{VARIANTS[v][1][:12]} -> {d}")
        return

    import mlx.core as mx
    import mlx.nn as nn
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    options = dict(
        language="en",
        task="transcribe",
        temperature=0.0,  # a float, not a tuple: one greedy decode, no fallback
        condition_on_previous_text=False,
        word_timestamps=False,
        initial_prompt=None,
        without_timestamps=not args.timestamps,
        fp16=True,
        verbose=None,  # no printing and no progress bar
    )

    path = str(model_dir(args.variant))
    repo, revision = VARIANTS[args.variant]
    info = mx.device_info()
    log(f"== {args.variant}: {repo}@{revision[:12]}")
    log(f"   python {sys.version.split()[0]}  mlx {mx.__version__}  mlx-whisper {mlx_whisper.__version__}"
        f"  device {mx.default_device()} {info.get('device_name')} ({info.get('architecture')})")
    log(f"   options {options}")

    t0 = time.perf_counter()
    model = ModelHolder.get_model(path, mx.float16)  # transcribe() reuses it: same path, same dtype
    load_ms = 1000 * (time.perf_counter() - t0)
    quant = [m for _, m in model.named_modules() if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
    kind = f"{len(quant)} layers quantized {quant[0].bits}-bit g{quant[0].group_size}" if quant else "fp16"
    log(f"   model {path}\n   loaded in {load_ms:.0f} ms ({kind})")

    def timed(audio: np.ndarray):
        gc.collect()  # outside the timed region
        start = time.perf_counter_ns()
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=path, **options)
        text = result["text"]
        mx.synchronize()
        ms = (time.perf_counter_ns() - start) / 1e6
        assert ModelHolder.model is model, "model was reloaded inside the timed region"
        return ms, clean(text)

    clips = {}
    for name in dict.fromkeys(args.warmup.split(",") + args.clips.split(",")):
        if name:
            clips[name] = load_16k_mono(AUDIO_DIR / f"{name}.wav")

    for name in filter(None, args.warmup.split(",")):
        ms, text = timed(clips[name])
        log(f"   warmup {name:6s} {ms:8.1f} ms  {text[:70]}")

    if args.latency:
        out = Writer(args.out_dir / f"{args.variant}-latency.tsv")
        for name in filter(None, args.clips.split(",")):
            for run in range(1, args.runs + 1):
                ms, text = timed(clips[name])
                out.row(name, run, f"{ms:.2f}", text)
                log(f"   {name:6s} run {run}  {ms:8.2f} ms  {text}")
        out.close()

    if args.eval:
        items = []
        for line in args.eval_list.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(eval_audio_path(line.split("\t", 1)[0], args.eval_list))
        audio = [load_16k_mono(p) for p in items]  # every file decoded before any timing
        log(f"   eval: {len(items)} files decoded ({sum(map(len, audio)) / SAMPLE_RATE:.1f} s of audio)")
        out = Writer(args.out_dir / f"{args.variant}-eval.tsv")
        total = 0.0
        for i, (p, a) in enumerate(zip(items, audio), 1):
            ms, text = timed(a)
            total += ms
            out.row(p.name, f"{ms:.2f}", text)
            if i % 20 == 0 or i == len(items):
                log(f"   eval {i}/{len(items)}  last {ms:.1f} ms  mean {total / i:.1f} ms")
        out.close()


if __name__ == "__main__":
    main()
