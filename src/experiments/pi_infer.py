"""
Live inference — using the model that was just trained
======================================================

The finale. Everything before this trains a classifier; this is where it
gets used. Runs on the Pi, listens through the microphone, and says what it
hears:

    every tick (~1s):
        record 500ms of audio -> STFT -> model.predict
        also run YOLO on a frame, purely for SIDE-BY-SIDE comparison
        print what the ear says, what the eye says, and whether they agree

Three behaviours worth demonstrating, in increasing order of interest:

1. IT WORKS.  Hold up a class it was trained on; the audio branch names it
   without the camera being involved in the decision at all.

2. IT KNOWS WHAT IT DOESN'T KNOW.  A softmax head is closed-set — without a
   reject option every sound is forced into a trained class, often
   confidently. Below --threshold this prints `unknown` instead of
   guessing. Make a sound it never learned and watch it decline to answer.

3. IT ASKS TO BE RETRAINED.  After --novelty-ticks consecutive unknowns
   while the camera CAN see an object, it reports a novel class and names
   what vision thinks it is. That closes the thesis loop: the device
   notices it is out of its depth, and federated learning is what makes
   acting on that cheap enough to be worth doing.

COVER THE CAMERA and the audio prediction carries on unchanged — that is
the "audio helps when vision fails" claim, demonstrated rather than
asserted.

NOTE — vision is OFF by default, and not by preference. On the Pi 5,
TensorFlow 2.21 and the PyTorch that ultralytics pulls in segfault when
loaded into the same process (an OpenMP clash on aarch64 that survives
both import reordering and OMP_NUM_THREADS=1). Enrollment uses YOLO with
no TensorFlow, and inference uses TensorFlow with no YOLO, so nothing in
the pipeline actually needs both at once. --with-vision is available for a
machine where they coexist.

Run on the Pi, after a strategy server has written a model:

    .venv/bin/python -m src.experiments.pi_infer \
        --model results/live/trained.keras --audio-device 0

Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pi_live import MicRecorder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Trained .keras written by a strategy server's "
                        "--save-model (its .classes.json sits alongside)")
    p.add_argument("--audio-device", type=int, default=None,
                   help="sounddevice input index (list: python -m sounddevice)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Below this top-probability the prediction is "
                        "reported as `unknown` rather than guessed. 0 "
                        "disables the reject and restores plain argmax.")
    p.add_argument("--novelty-ticks", type=int, default=5,
                   help="Consecutive unknowns before flagging a novel class")
    p.add_argument("--tick-ms", type=int, default=1000)
    p.add_argument("--with-vision", action="store_true",
                   help="ALSO run YOLO for a side-by-side column. WARNING: "
                        "on this Pi, TensorFlow and PyTorch segfault when "
                        "loaded in one process (an OpenMP clash on aarch64 "
                        "that survives import reordering and "
                        "OMP_NUM_THREADS). Off by default; audio-only is "
                        "both the working path and the deployment case the "
                        "thesis argues for.")
    p.add_argument("--yolo-model", default="yolov8n.pt")
    p.add_argument("--duration-min", type=float, default=None)
    p.add_argument("--run-dir", default="results/live/inference")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from src.edge.processing.stft import STFTProcessor

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: no model at {model_path}\n"
              "  A strategy server writes one when given --save-model.",
              file=sys.stderr)
        return 1

    sidecar = model_path.with_suffix(".classes.json")
    if not sidecar.exists():
        print(f"ERROR: missing {sidecar}\n"
              "  The .keras records the head width but not what the outputs "
              "mean; the sidecar carries the class names.", file=sys.stderr)
        return 1
    classes = json.loads(sidecar.read_text())["classes"]

    print(f"[infer] loading {model_path.name} ...", flush=True)
    model = tf.keras.models.load_model(model_path)
    print(f"[infer] classes: {classes}", flush=True)

    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)

    mic = MicRecorder(args.audio_device)
    print(f"[infer] mic: {mic.name} @ {mic.native_sr} Hz -> 16000 Hz",
          flush=True)

    vision = None
    cam = None
    if args.with_vision:
        import cv2
        from src.edge.processing.vision import VisionProcessor
        cam = cv2.VideoCapture(args.camera_index)
        if not cam.isOpened():
            print("WARN: cannot open camera — continuing audio-only",
                  flush=True)
            cam = None
        else:
            vision = VisionProcessor(model_path=args.yolo_model,
                                     confidence_threshold=0.4)
            ok, warm = cam.read()
            if ok:
                vision.detect(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB))
            print("[infer] YOLO warmed up (shown for comparison only — it "
                  "does not feed the audio prediction)", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    unknown_streak = 0
    novelty_flagged = False
    recent_vision: deque = deque(maxlen=args.novelty_ticks)
    heard: Counter = Counter()
    ticks = 0
    agreements = 0
    comparable = 0

    print(f"\n{'EAR (audio model)':<28}{'EYE (YOLO)':<22}note")
    print("-" * 72, flush=True)

    t_start = time.perf_counter()
    deadline = (t_start + args.duration_min * 60) if args.duration_min else None

    while not stop["flag"]:
        if deadline and time.perf_counter() > deadline:
            break
        tick_t0 = time.perf_counter()
        ticks += 1

        rec = mic.start_chunk()
        frame_bgr = None
        if cam is not None:
            ok, frame_bgr = cam.read()
            if not ok:
                frame_bgr = None
        audio = mic.finish(rec)

        spec = stft.process(audio)[np.newaxis, ..., np.newaxis].astype(np.float32)
        probs = model.predict(spec, verbose=0)[0]
        top = int(probs.argmax())
        conf = float(probs[top])
        known = conf >= args.threshold
        label = classes[top] if known else "unknown"
        heard[label] += 1

        seen = "-"
        if vision is not None and frame_bgr is not None:
            import cv2
            dets = vision.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            picked = vision.get_dominant_label(dets)
            if picked:
                seen = picked[0]
        recent_vision.append(seen)

        note = ""
        if known:
            unknown_streak = 0
            novelty_flagged = False
            if seen != "-":
                comparable += 1
                if seen == label:
                    agreements += 1
                    note = "agree"
                else:
                    note = "DISAGREE — audio and vision differ"
        else:
            unknown_streak += 1
            note = f"below {args.threshold:.2f} — refusing to guess"
            if unknown_streak >= args.novelty_ticks and not novelty_flagged:
                # Something is consistently audible that the model cannot
                # name. If vision can see it, we even know what to call it.
                candidates = [v for v in recent_vision if v != "-"]
                guess = Counter(candidates).most_common(1)
                novelty_flagged = True
                print("-" * 72, flush=True)
                if guess and guess[0][1] >= 2:
                    print(f"  NEW CLASS: {unknown_streak} unknowns in a row "
                          f"while the camera sees '{guess[0][0]}'.\n"
                          f"  Not in {classes} — enroll it and retrain.",
                          flush=True)
                else:
                    print(f"  NEW SOUND: {unknown_streak} unknowns in a row, "
                          f"nothing recognisable in view.\n"
                          f"  Out of vocabulary {classes}.", flush=True)
                print("-" * 72, flush=True)

        bar = "#" * int(conf * 20)
        print(f"{label + f' {conf:.2f}':<28}{seen:<22}{note}  {bar}",
              flush=True)

        delay = args.tick_ms / 1000.0 - (time.perf_counter() - tick_t0)
        if delay > 0:
            time.sleep(delay)

    if cam is not None:
        cam.release()

    result = {
        "model": str(model_path),
        "classes": classes,
        "threshold": args.threshold,
        "ticks": ticks,
        "heard": dict(heard),
        "unknown_rate": round(heard.get("unknown", 0) / max(ticks, 1), 3),
        "vision_comparable_ticks": comparable,
        "audio_vision_agreement": (round(agreements / comparable, 3)
                                   if comparable else None),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    out = run_dir / "inference.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\n--- {ticks} ticks ---")
    for k, v in heard.most_common():
        print(f"  {k:<20} {v:4d}")
    if comparable:
        print(f"  audio/vision agreement: {result['audio_vision_agreement']:.0%} "
              f"over {comparable} ticks where both saw something")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
