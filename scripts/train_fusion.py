"""
F3: train FusionModel + its two baselines on the VGGSound paired cache
======================================================================

Trains three models on the SAME data (data/vggsound/paired_cache.npz):

    fusion  — image + audio (FusionModel: frozen MobileNetV3 + AudioCNN + head)
    audio   — audio-only baseline (AudioOnlyModel)
    vision  — vision-only baseline (frozen MobileNetV3 + head)

The two baselines are what make a fusion claim meaningful: "fusion X% vs
audio-only Y% vs vision-only Z%".

F4 hook: --modality-dropout P randomly blanks the IMAGE on a fraction P of
training samples (fusion only), forcing audio competence — the training
trick behind "still works when vision fails" (F5).

Outputs:
    models/fusion_{fusion,audio,vision}.keras
    data/vggsound/fusion_training_metrics.json

Usage:
    .venv/bin/python scripts/train_fusion.py                 # all three
    .venv/bin/python scripts/train_fusion.py --model fusion --modality-dropout 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, Model  # noqa: E402

from src.server.training.models.fusion_model import (  # noqa: E402
    FusionModel, AudioOnlyModel)

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
MODELS_DIR = REPO_ROOT / "models"
METRICS_OUT = REPO_ROOT / "data/vggsound/fusion_training_metrics.json"

SEED = 42
BATCH = 32
EPOCHS = 40
LR = 1e-3
ES_PATIENCE = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["fusion", "audio", "vision", "all"],
                   default="all")
    p.add_argument("--modality-dropout", type=float, default=0.0,
                   help="Fraction of TRAIN samples whose image is blanked "
                        "(fusion only). 0 = off (F3), e.g. 0.3 for F4.")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--audio-warmstart", action="store_true",
                   help="Initialize the fusion audio branch from the trained "
                        "audio-only model (models/fusion_audio.keras). Cures "
                        "modality imbalance: a from-scratch audio branch gets "
                        "starved of gradient next to the pretrained vision "
                        "branch and ends up ignored by the head.")
    p.add_argument("--es-patience", type=int, default=ES_PATIENCE,
                   help="EarlyStopping patience; >= --epochs disables it "
                        "(clean-val ES can cut training before the audio "
                        "pathway forms under modality dropout)")
    return p.parse_args()


def VisionOnlyModel(num_classes: int) -> Model:
    """Frozen MobileNetV3-Small + small head — the vision baseline."""
    inp = layers.Input(shape=(224, 224, 3), name="vision_input")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3), include_top=False,
        weights="imagenet", pooling="avg")
    backbone.trainable = False
    x = backbone(inp)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="vision_only_model")


def make_dataset(images, img_idx, X_aud, y, kind: str, mode: str,
                 modality_dropout: float, shuffle: bool) -> tf.data.Dataset:
    """kind: which inputs the model wants. Images gathered by index on the
    fly (stored once per clip in the cache), cast to float32 0-255 —
    Keras MobileNetV3 rescales internally."""
    images_t = tf.constant(images)  # (n_clips,224,224,3) uint8
    ds = tf.data.Dataset.from_tensor_slices((img_idx, X_aud, y))
    if shuffle:
        ds = ds.shuffle(len(y), seed=SEED)

    def _map(idx, aud, label):
        img = tf.cast(tf.gather(images_t, idx), tf.float32)
        if kind == "fusion":
            if mode == "train" and modality_dropout > 0:
                # F4: blank the image on a random fraction of samples so the
                # fusion head learns audio-only competence.
                # NOTE: no op-level seed here — a fixed seed inside tf.data
                # yields the SAME draw for every element (dropout never fires).
                # Global tf.random.set_seed keeps runs reproducible.
                drop = tf.random.uniform([]) < modality_dropout
                img = tf.cond(drop, lambda: tf.zeros_like(img), lambda: img)
            return {"vision_input": img, "audio_input": aud}, label
        if kind == "vision":
            return img, label
        return aud, label  # audio

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(BATCH).prefetch(tf.data.AUTOTUNE)


def class_weights(y: np.ndarray, n: int) -> dict[int, float]:
    counts = np.bincount(y, minlength=n).astype(np.float64)
    counts[counts == 0] = 1.0
    return {i: float(len(y) / (n * c)) for i, c in enumerate(counts)}


def per_class_accuracy(model, ds, y_true: np.ndarray, classes) -> dict:
    y_prob = model.predict(ds, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    out = {}
    for i, c in enumerate(classes):
        mask = y_true == i
        out[c] = round(float((y_pred[mask] == i).mean()), 3) if mask.any() else None
    return out


def main() -> int:
    args = parse_args()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not CACHE.exists():
        print(f"ERROR: {CACHE} missing — run prepare_vggsound_pairs.py",
              file=sys.stderr)
        return 2

    d = np.load(CACHE, allow_pickle=True)
    images = d["images"]
    classes = [str(c) for c in d["classes"]]
    n_cls = len(classes)
    print(f"cache: {images.shape[0]} clips, classes={classes}")
    splits = {s: (d[f"img_idx_{s}"], d[f"X_aud_{s}"], d[f"y_{s}"])
              for s in ("train", "val", "test")}
    for s, (_, Xa, y) in splits.items():
        print(f"  {s}: {len(y)} samples")

    builders = {
        "fusion": lambda: FusionModel(num_classes=n_cls),
        "audio":  lambda: AudioOnlyModel(num_classes=n_cls),
        "vision": lambda: VisionOnlyModel(num_classes=n_cls),
    }
    targets = list(builders) if args.model == "all" else [args.model]

    cw = class_weights(splits["train"][2], n_cls)
    results = {}
    MODELS_DIR.mkdir(exist_ok=True)

    for name in targets:
        # A dropout run is a distinct result — don't overwrite the naive one
        result_key = (f"fusion_md{args.modality_dropout}"
                      if name == "fusion" and args.modality_dropout > 0 else name)
        print(f"\n===== {result_key} =====", flush=True)
        model = builders[name]()
        if name == "fusion" and args.audio_warmstart:
            donor = tf.keras.models.load_model(MODELS_DIR / "fusion_audio.keras")
            # Fusion's audio branch = the only top-level Conv2D/BN layers
            # (MobileNet is a nested Model); donor has the identical stack.
            def stack(m, types):
                return [l for l in m.layers if isinstance(l, types)]
            conv_bn = (tf.keras.layers.Conv2D, tf.keras.layers.BatchNormalization)
            for src, dst in zip(stack(donor, conv_bn), stack(model, conv_bn)):
                dst.set_weights(src.get_weights())
            # donor's first Dense(128) → fusion's audio_projection
            donor_dense = stack(donor, (tf.keras.layers.Dense,))[0]
            model.get_layer("audio_projection").set_weights(donor_dense.get_weights())
            n = len(stack(donor, conv_bn)) + 1
            print(f"audio branch warm-started from fusion_audio.keras "
                  f"({n} layers transferred)", flush=True)
            result_key += "_aws"
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        md = args.modality_dropout if name == "fusion" else 0.0
        ds_tr = make_dataset(images, *splits["train"], kind=name, mode="train",
                             modality_dropout=md, shuffle=True)
        ds_va = make_dataset(images, *splits["val"], kind=name, mode="eval",
                             modality_dropout=0, shuffle=False)
        ds_te = make_dataset(images, *splits["test"], kind=name, mode="eval",
                             modality_dropout=0, shuffle=False)

        t0 = time.perf_counter()
        hist = model.fit(
            ds_tr, validation_data=ds_va, epochs=args.epochs,
            class_weight=cw, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.es_patience,
                restore_best_weights=True, verbose=1)])
        train_s = time.perf_counter() - t0

        _, test_acc = model.evaluate(ds_te, verbose=0)
        pca = per_class_accuracy(model, ds_te, splits["test"][2], classes)
        out_path = MODELS_DIR / f"fusion_{result_key}.keras"
        model.save(out_path)
        results[result_key] = {
            "test_accuracy": round(float(test_acc), 4),
            "per_class_test_accuracy": pca,
            "epochs_run": len(hist.history["loss"]),
            "train_seconds": round(train_s, 1),
            "modality_dropout": md,
            "model_path": str(out_path.relative_to(REPO_ROOT)),
        }
        print(f"{result_key}: test_acc={test_acc:.3f} "
              f"({len(hist.history['loss'])} epochs, {train_s:.0f}s) "
              f"per-class={pca}", flush=True)

    payload = {"classes": classes,
               "train_samples": int(len(splits["train"][2])),
               "seed": SEED, "results": results}
    if METRICS_OUT.exists():  # merge with earlier partial runs
        old = json.loads(METRICS_OUT.read_text())
        old.get("results", {}).update(results)
        payload["results"] = old["results"]
    METRICS_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved metrics: {METRICS_OUT}")
    print(json.dumps({k: v["test_accuracy"] for k, v in payload["results"].items()},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
