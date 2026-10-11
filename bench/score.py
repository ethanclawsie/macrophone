"""Scores every bench/results/*-eval.tsv with Whisper's English text normalizer (the Open ASR
Leaderboard's), so formatting differences — "2000" vs "two thousand", "luster" vs "lustre", casing,
punctuation — don't count as recognition errors.

WER is reported over all utterances and over those up to 30 s (one Whisper window): the two longer
ones also test how each configuration continues past the first window.

    uv run --no-project --with whisper-normalizer==0.1.15 python -I bench/score.py
"""
import glob
import os
import statistics
import sys

from whisper_normalizer.english import EnglishTextNormalizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
normalize = EnglishTextNormalizer()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


refs = {}
with open(os.path.join(ROOT, "data", "eval-160.tsv")) as f:
    for line in f:
        path, seconds, ref = line.rstrip("\n").split("\t", 2)
        refs[os.path.basename(path)] = (float(seconds), normalize(ref).split())

buckets = [("<3s", 0, 3), ("3-6s", 3, 6), ("6-12s", 6, 12), ("12s+", 12, 1e9)]
rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "results", "*-eval.tsv"))):
    name = os.path.basename(path)[: -len("-eval.tsv")]
    errs = {b[0]: [0, 0] for b in buckets}
    short = [0, 0]  # utterances up to 30 s
    lat = []
    seen = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 2 or parts[0] not in refs:
                continue
            seconds, ref = refs[parts[0]]
            hyp = normalize(parts[2] if len(parts) > 2 else "").split()
            b = next(b[0] for b in buckets if b[1] <= seconds < b[2])
            e = edit_distance(ref, hyp)
            errs[b][0] += e
            errs[b][1] += len(ref)
            if seconds <= 30:
                short[0] += e
                short[1] += len(ref)
            lat.append(float(parts[1]))
            seen += 1
    total_e = sum(e for e, _ in errs.values())
    total_w = sum(w for _, w in errs.values())
    per_bucket = " ".join(f"{100 * e / max(w, 1):5.2f}" for e, w in errs.values())
    rows.append((statistics.median(lat), name, 100 * total_e / max(total_w, 1),
                 100 * short[0] / max(short[1], 1), per_bucket, seen))

print(f"{'implementation':34s} {'WER':>6s} {'<=30s':>6s}   {'<3s':>5s} {'3-6s':>5s} {'6-12':>5s} {'12s+':>5s}   {'median ms':>9s}  n")
for med, name, wer, wer_short, per_bucket, seen in sorted(rows):
    print(f"{name:34s} {wer:5.2f}% {wer_short:5.2f}%   {per_bucket}   {med:9.0f}  {seen}")
