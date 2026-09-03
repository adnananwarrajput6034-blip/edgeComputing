"""
Live enrollment — the prof picks the classes
============================================

Step 1 of the live demo track. Runs on the Pi, captures real audio, and
lets YOLO name it:

    every tick (~1s):
        start 500ms mic recording
        grab a frame mid-recording (image sits inside the audio window)
        YOLO labels the frame -> that name labels the audio
        nothing detected -> tick skipped, never guessed

No vocabulary is fixed in advance. Show a keyboard and type, show scissors
and snip: the classes are whatever objects were actually detected. This is
the thesis's weak-supervision loop running literally — vision supervises
audio, with no human labelling anywhere in the path.

Classes below --min-samples are dropped when the vocabulary is frozen, so a
passer-by detected twice does not become a class.

Run on the Pi:

    .venv/bin/python -m src.experiments.pi_enroll \
        --audio-device 0 --min-samples 30 --save-evidence

Ctrl-C when the readout shows enough samples per class. Writes
{--out} (default results/live/live_dataset.npz), which is then copied to
the laptop and replayed through strategies A, B and C.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pi_live import enroll, save_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/live/live_dataset.npz",
                   help="Where to write the enrolled dataset")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--audio-device", type=int, default=None,
                   help="sounddevice input index (list: python -m sounddevice)")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="YOLO detection threshold; below this the tick is skipped")
    p.add_argument("--min-samples", type=int, default=30,
                   help="Classes with fewer samples are dropped from the "
                        "vocabulary (a passer-by is not a class)")
    p.add_argument("--ignore", default="",
                   help="Comma-separated labels to ignore, e.g. "
                        "'person,tv'. YOLO picks the highest-confidence "
                        "detection, so the operator standing in frame "
                        "outscores the object being enrolled — ignore "
                        "'person' and the object wins instead.")
    p.add_argument("--tick-ms", type=int, default=1000)
    p.add_argument("--duration-min", type=float, default=None,
                   help="Stop after N minutes (default: run until Ctrl-C)")
    p.add_argument("--test-fraction", type=float, default=0.25,
                   help="Held-out fraction, stratified per class")
    p.add_argument("--save-evidence", action="store_true",
                   help="DEMO ONLY: also save frame.jpg + audio.wav per "
                        "labeled tick so a human can verify what YOLO saw. "
                        "The production pipeline never persists frames.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    evidence_dir = None
    if args.save_evidence:
        evidence_dir = out.parent / f"evidence_{time.strftime('%Y%m%d_%H%M%S')}"
        evidence_dir.mkdir(parents=True)
        print(f"[enroll] EVIDENCE MODE -> {evidence_dir}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    audio, labels, stats = enroll(
        camera_index=args.camera_index,
        audio_device=args.audio_device,
        yolo_model=args.model,
        confidence=args.confidence,
        min_samples=args.min_samples,
        duration_min=args.duration_min,
        tick_ms=args.tick_ms,
        stop_flag=stop,
        evidence_dir=evidence_dir,
        ignore=tuple(args.ignore.split(",")) if args.ignore else (),
    )

    if not labels:
        print("\nERROR: nothing was detected — no samples enrolled.\n"
              "  Check the camera is pointed at the object and that YOLO\n"
              "  recognises it (try --confidence 0.3).", file=sys.stderr)
        return 1

    # ---- freeze the vocabulary -------------------------------------------
    counts = stats["raw_counts"]
    classes = sorted(c for c, n in counts.items() if n >= args.min_samples)
    dropped = {c: n for c, n in counts.items() if n < args.min_samples}

    print("\n--- enrollment summary ---")
    print(f"ticks {stats['ticks']}, labeled {stats['labeled']} "
          f"({stats['detection_rate']:.0%}), "
          f"YOLO {stats['yolo_ms_mean']} ms/frame")
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        mark = "KEPT   " if c in classes else "dropped"
        print(f"  {mark} {c:<16s} {n:4d} samples")
    if dropped:
        print(f"  (dropped: below --min-samples {args.min_samples})")

    if len(classes) < 2:
        print(f"\nERROR: need >=2 classes with >={args.min_samples} samples, "
              f"got {len(classes)}.\n"
              "  Enroll for longer, add another object, or lower "
              "--min-samples.", file=sys.stderr)
        return 1

    keep = [i for i, l in enumerate(labels) if l in classes]
    audio = audio[keep]
    labels = [labels[i] for i in keep]

    info = save_dataset(out, audio, labels, classes,
                        test_fraction=args.test_fraction)

    meta = {**stats, "classes": classes, "dropped_classes": dropped,
            "min_samples": args.min_samples, **info}
    (out.parent / "enroll_report.json").write_text(json.dumps(meta, indent=2))

    if evidence_dir is not None:
        (evidence_dir / "manifest.json").write_text(json.dumps(meta, indent=2))

    print(f"\nvocabulary: {classes}")
    print(f"train {info['n_train']} / test {info['n_test']} samples")
    print(f"wrote {out}")
    print("\nNext: copy to the laptop, then run strategies A/B/C against it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
