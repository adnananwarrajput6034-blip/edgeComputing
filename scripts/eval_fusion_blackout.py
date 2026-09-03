"""
F5: "audio helps when vision fails" — blackout evaluation
=========================================================

No training. Evaluates every saved fusion-track model on the SAME test set
under two conditions:

    clean     — normal frames + audio
    blackout  — every frame replaced by black (fog / darkness / dead camera);
                audio unchanged

Models compared (from models/fusion_*.keras):
    vision        — should COLLAPSE to ~chance under blackout
    fusion        — naive fusion; expected to degrade badly (F3 lesson)
    fusion_md0.3  — modality-dropout fusion; should HOLD near the audio floor
    audio         — unaffected by blackout (the reference floor)

Output: data/vggsound/fusion_blackout_results.json + printed table.

Usage:
    .venv/bin/python scripts/eval_fusion_blackout.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
MODELS_DIR = REPO_ROOT / "models"
OUT = REPO_ROOT / "data/vggsound/fusion_blackout_results.json"
BATCH = 32


def main() -> int:
    d = np.load(CACHE, allow_pickle=True)
    images = d["images"]
    classes = [str(c) for c in d["classes"]]
    idx, X_aud, y = d["img_idx_test"], d["X_aud_test"], d["y_test"]
    X_img = images[idx].astype(np.float32)          # clean frames
    X_blk = np.zeros_like(X_img)                    # blackout frames
    n = len(y)
    chance = 1.0 / len(classes)
    print(f"test set: {n} samples, {len(classes)} classes "
          f"(chance = {chance:.2f})\n")

    model_files = sorted(MODELS_DIR.glob("fusion_*.keras"))
    if not model_files:
        print("ERROR: no models/fusion_*.keras — run train_fusion.py first",
              file=sys.stderr)
        return 2

    def acc(model, inputs) -> float:
        prob = model.predict(inputs, batch_size=BATCH, verbose=0)
        return float((prob.argmax(axis=1) == y).mean())

    results = {}
    for path in model_files:
        name = path.stem.removeprefix("fusion_")
        model = tf.keras.models.load_model(path)
        n_inputs = len(model.inputs)
        if n_inputs == 2:            # fusion variants
            clean = acc(model, {"vision_input": X_img, "audio_input": X_aud})
            blackout = acc(model, {"vision_input": X_blk, "audio_input": X_aud})
        elif "vision" in name:       # vision-only
            clean = acc(model, X_img)
            blackout = acc(model, X_blk)
        else:                        # audio-only: blackout is irrelevant
            clean = acc(model, X_aud)
            blackout = clean
        results[name] = {"clean": round(clean, 4), "blackout": round(blackout, 4),
                         "degradation": round(clean - blackout, 4)}
        print(f"{name:14s} clean={clean:.3f}  blackout={blackout:.3f}  "
              f"drop={clean - blackout:+.3f}")

    OUT.write_text(json.dumps(
        {"classes": classes, "test_samples": int(n), "chance": round(chance, 3),
         "results": results}, indent=2))
    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
