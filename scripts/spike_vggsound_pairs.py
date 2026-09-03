"""
F1 spike: can we harvest paired (frame, audio, label) data from VGGSound?
=========================================================================

The fusion model needs training examples with BOTH modalities: an image and
its simultaneous audio, plus a label. UrbanSound8K has no video. This spike
measures whether VGGSound (YouTube clips + human sound-labels) can supply
paired data at usable volume — BEFORE we build the full F2 pipeline on it.

Per clip: yt-dlp downloads a 10 s low-res segment at the labeled timestamp,
ffmpeg extracts the middle frame (jpg) + 16 kHz mono audio (wav).

Outputs:
    data/vggsound/pairs/{coco_class}/{yt_id}.jpg + .wav
    data/vggsound/pairs_manifest.json      (kept samples + fail log + stats)

Success criteria for the spike:
    - overall success rate ≥ ~50% of attempts (link rot is expected)
    - time per kept clip small enough to extrapolate to hundreds/class

Usage:
    .venv/bin/python scripts/spike_vggsound_pairs.py --per-class 8
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YTDLP = REPO_ROOT / ".venv" / "bin" / "yt-dlp"

# Same class mapping the label-noise study used: VGGSound label -> COCO class
# (all five are objects YOLO can see AND that make characteristic sounds).
TARGET_CLASSES = {
    "car passing by":          "car",
    "driving motorcycle":      "motorcycle",
    "dog barking":             "dog",
    "cat meowing":             "cat",
    "bird chirping, tweeting": "bird",
}

CSV_PATH = REPO_ROOT / "data/vggsound/vggsound.csv"
OUT_ROOT = REPO_ROOT / "data/vggsound/pairs"
MANIFEST = REPO_ROOT / "data/vggsound/pairs_manifest.json"

CLIP_SECONDS = 10  # VGGSound labels refer to a 10 s window from `start`


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=8,
                   help="Paired samples to keep per class")
    p.add_argument("--max-attempts-per-class", type=int, default=24,
                   help="Give up on a class after this many tries")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=90.0,
                   help="Seconds allowed per clip download")
    return p.parse_args()


def download_segment(yt_id: str, start: int, dest_mp4: Path, timeout: float) -> str | None:
    """Download a 10s low-res segment. Returns None on success, else fail reason."""
    end = start + CLIP_SECONDS
    cmd = [
        str(YTDLP),
        "--quiet", "--no-warnings",
        "-f", "b[height<=360]/w",          # small; we only need one frame + audio
        "--download-sections", f"*{start}-{end}",
        "--no-playlist",
        "--force-overwrites",
        "-o", str(dest_mp4),
        f"https://www.youtube.com/watch?v={yt_id}",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout"
    if res.returncode != 0:
        err = (res.stderr or "").lower()
        for marker, reason in [
            ("video unavailable", "unavailable"),
            ("private video", "private"),
            ("removed", "removed"),
            ("age", "age_restricted"),
            ("sign in", "login_required"),
        ]:
            if marker in err:
                return reason
        return "yt-dlp_error"
    if not dest_mp4.exists() or dest_mp4.stat().st_size < 10_000:
        return "empty_download"
    return None


def extract_pair(mp4: Path, jpg: Path, wav: Path) -> str | None:
    """Middle frame + 16 kHz mono audio from the segment. None on success."""
    mid = CLIP_SECONDS / 2
    f = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(mid), "-i", str(mp4),
         "-frames:v", "1", "-q:v", "2", str(jpg)],
        capture_output=True, text=True)
    if f.returncode != 0 or not jpg.exists():
        return "frame_extract_failed"
    a = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        capture_output=True, text=True)
    if a.returncode != 0 or not wav.exists() or wav.stat().st_size < 32_000:
        return "audio_extract_failed"  # <1s of 16k int16 ≈ silence/broken
    return None


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} missing", file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not on PATH", file=sys.stderr)
        return 2

    args = parse_args()
    rng = random.Random(args.seed)

    by_class: dict[str, list[tuple[str, int]]] = {c: [] for c in TARGET_CLASSES}
    with CSV_PATH.open() as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[2] in by_class:
                by_class[row[2]].append((row[0], int(row[1])))

    print("Candidates per class:")
    for c, items in by_class.items():
        print(f"  {len(items):>5}  {c}")

    kept_all: list[dict] = []
    failed_all: list[dict] = []
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmp:
        for vgg_cls, coco_cls in TARGET_CLASSES.items():
            out_dir = OUT_ROOT / coco_cls
            out_dir.mkdir(parents=True, exist_ok=True)
            pool = by_class[vgg_cls][:]
            rng.shuffle(pool)

            kept = attempts = 0
            t0 = time.perf_counter()
            for yt_id, start in pool:
                if kept >= args.per_class or attempts >= args.max_attempts_per_class:
                    break
                # Resume: pair already harvested in a previous run → keep, no attempt
                jpg = out_dir / f"{yt_id}.jpg"
                wav = out_dir / f"{yt_id}.wav"
                if jpg.exists() and wav.exists():
                    kept += 1
                    kept_all.append({"yt_id": yt_id, "start": start,
                                     "vgg_class": vgg_cls, "coco_class": coco_cls,
                                     "frame": str(jpg.relative_to(REPO_ROOT)),
                                     "audio": str(wav.relative_to(REPO_ROOT))})
                    continue
                attempts += 1
                mp4 = Path(tmp) / f"{yt_id}.mp4"
                reason = download_segment(yt_id, start, mp4, args.timeout)
                if reason is None:
                    jpg = out_dir / f"{yt_id}.jpg"
                    wav = out_dir / f"{yt_id}.wav"
                    reason = extract_pair(mp4, jpg, wav)
                    mp4.unlink(missing_ok=True)
                if reason is None:
                    kept += 1
                    kept_all.append({"yt_id": yt_id, "start": start,
                                     "vgg_class": vgg_cls, "coco_class": coco_cls,
                                     "frame": str(jpg.relative_to(REPO_ROOT)),
                                     "audio": str(wav.relative_to(REPO_ROOT))})
                    print(f"  [{coco_cls}] ok  {yt_id}  ({kept}/{args.per_class})",
                          flush=True)
                else:
                    failed_all.append({"yt_id": yt_id, "vgg_class": vgg_cls,
                                       "reason": reason})
                    print(f"  [{coco_cls}] FAIL {yt_id}: {reason}", flush=True)

            print(f"{coco_cls}: kept {kept}/{args.per_class} in {attempts} attempts "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)

    wall = time.perf_counter() - t_start
    n_ok, n_fail = len(kept_all), len(failed_all)
    reasons: dict[str, int] = {}
    for f_ in failed_all:
        reasons[f_["reason"]] = reasons.get(f_["reason"], 0) + 1

    stats = {
        "kept": n_ok, "failed": n_fail,
        "success_rate": round(n_ok / max(n_ok + n_fail, 1), 3),
        "fail_reasons": reasons,
        "wall_seconds": round(wall, 1),
        "seconds_per_kept": round(wall / max(n_ok, 1), 1),
        "per_class": {c: sum(1 for k in kept_all if k["coco_class"] == c)
                      for c in set(TARGET_CLASSES.values())},
    }
    MANIFEST.write_text(json.dumps(
        {"samples": kept_all, "failed": failed_all, "stats": stats}, indent=2))

    print(f"\n=== SPIKE RESULT ===")
    print(json.dumps(stats, indent=2))
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
