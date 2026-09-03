"""
F6 step 1: precompute vision feature vectors + build the FV-fusion model
========================================================================

The F6 wire format ships the frozen MobileNetV3 backbone's 576-float output
(the "vision FV") instead of the frame. This script:

1. Runs the frozen backbone over every cached clip frame → fv per clip,
   saved to data/vggsound/fv_cache.npz (aligned with paired_cache images).
   On a real edge device this happens per capture; precomputing here is the
   simulation equivalent (the per-frame cost is measured in F7).

2. Builds FVFusionModel, transfers ALL trained weights from the proven
   image-input model (models/fusion_fusion_md0.5_aws.keras), and VERIFIES
   equivalence: FV-model(backbone(img), audio) must equal image-model(img,
   audio) on the test set, clean and blackout. Saves models/fv_fusion.keras.

The equivalence check is the point: it proves the edge/server split changes
WHERE computation runs, not WHAT is computed.

Usage:
    .venv/bin/python scripts/prepare_fusion_fv.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

from src.server.training.models.fusion_model import (  # noqa: E402
    FVFusionModel, transfer_fusion_weights)

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
FV_OUT = REPO_ROOT / "data/vggsound/fv_cache.npz"
DONOR = REPO_ROOT / "models/fusion_fusion_md0.5_aws.keras"
MODEL_OUT = REPO_ROOT / "models/fv_fusion.keras"


def main() -> int:
    d = np.load(CACHE, allow_pickle=True)
    images = d["images"]
    classes = [str(c) for c in d["classes"]]

    # 1) precompute FVs for every clip frame
    print(f"Computing FVs for {len(images)} clip frames...", flush=True)
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False,
        weights="imagenet", pooling="avg")
    backbone.trainable = False
    t0 = time.perf_counter()
    fv = backbone.predict(images.astype(np.float32), batch_size=32, verbose=0)
    per_frame_ms = (time.perf_counter() - t0) / len(images) * 1000
    # the blackout FV: what the backbone emits for a pure black frame
    fv_black = backbone.predict(
        np.zeros((1, 224, 224, 3), np.float32), verbose=0)[0]
    np.savez_compressed(FV_OUT, fv=fv.astype(np.float32),
                        fv_black=fv_black.astype(np.float32))
    print(f"  fv: {fv.shape} → {FV_OUT} "
          f"({per_frame_ms:.1f} ms/frame on this machine)")

    # 2) FV-model + weight transfer + equivalence check
    if not DONOR.exists():
        print(f"ERROR: donor model {DONOR} missing — run train_fusion.py "
              f"--audio-warmstart first", file=sys.stderr)
        return 2
    donor = tf.keras.models.load_model(DONOR)
    fv_model = FVFusionModel(num_classes=len(classes))
    n = transfer_fusion_weights(donor, fv_model)
    print(f"  transferred {n} layers from {DONOR.name}")

    idx, X_aud, y = d["img_idx_test"], d["X_aud_test"], d["y_test"]
    X_img = images[idx].astype(np.float32)
    X_fv = fv[idx]
    X_fv_blk = np.tile(fv_black, (len(y), 1))
    X_blk = np.zeros_like(X_img)

    def acc(p):
        return float((p.argmax(1) == y).mean())

    for cond, img_in, fv_in in [("clean", X_img, X_fv),
                                ("blackout", X_blk, X_fv_blk)]:
        p_img = donor.predict(
            {"vision_input": img_in, "audio_input": X_aud}, verbose=0)
        p_fv = fv_model.predict(
            {"vision_fv_input": fv_in, "audio_input": X_aud}, verbose=0)
        agree = float((p_img.argmax(1) == p_fv.argmax(1)).mean())
        print(f"  {cond:9s} image-model acc={acc(p_img):.3f}  "
              f"fv-model acc={acc(p_fv):.3f}  prediction agreement={agree:.1%}")
        if agree < 0.99:
            print("  ERROR: FV split is NOT equivalent — investigate before F6",
                  file=sys.stderr)
            return 1

    fv_model.save(MODEL_OUT)
    print(f"Saved {MODEL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
