"""
Strategy C Client — Federated (local training, ships weights)
=============================================================

The privacy-preserving edge: data NEVER leaves the device. Each round the
client receives the global model, trains it on its OWN local data, and
ships back only the weights.

    receive global model (round r) → set_weights → train E local epochs
    on local data → publish weights + n_samples → wait for next global

Local data = this node's disjoint half of UrbanSound8K train folds, STFT'd
on-device (same as B), same deterministic split as A/B so the comparison is
apples-to-apples. Uses model.fit — the same training call the A/B servers
used — so C's local training is mechanically identical to A/B's server
training; only the LOCATION differs.

Round-synchronized with the server via retained global broadcasts; exits
when the server sets done=True.

Spawned by scripts/run_strategy.py --strategy c. Output:
{run-dir}/client_{node_id}.json with per-round LOCAL train time + upload bytes.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import queue
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import librosa  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.server.training.models.audio_cnn import AudioCNN  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000
TRAIN_FOLDS = set(range(1, 9))
MANIFEST_SEED = 42
GLOBAL_TOPIC = "thesis/server/model/global"
LR = 1e-3

METADATA = REPO_ROOT / "data/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_ROOT = REPO_ROOT / "data/urbansound8k/UrbanSound8K/audio"
CACHE = REPO_ROOT / "data/urbansound8k/cache.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Cap on local samples (capped again at manifest size)")
    p.add_argument("--stream-rate-ms", type=int, default=500,
                   help="accepted for harness compatibility (unused: C trains "
                        "on its full local set per round)")
    p.add_argument("--buffer-size", type=int, default=500,
                   help="accepted for harness compatibility (unused)")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def serialize_weights(weights: list, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf, meta=json.dumps(meta),
        **{f"w{i}": w for i, w in enumerate(weights)})
    return buf.getvalue()


def deserialize_weights(raw: bytes):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = sum(1 for k in z.files if k.startswith("w"))
    return [z[f"w{i}"] for i in range(n)], meta


def build_manifest(node_id: str) -> list[dict]:
    with METADATA.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["fold"]) in TRAIN_FOLDS]
    rows.sort(key=lambda r: (int(r["fold"]), r["slice_file_name"]))
    random.Random(MANIFEST_SEED).shuffle(rows)
    if node_id not in ("A", "B"):
        raise SystemExit(f"unsupported --node-id {node_id!r}")
    return rows[(0 if node_id == "A" else 1)::2]


def load_chunk(wav_path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    n = len(audio)
    if n >= CHUNK_SAMPLES:
        s = (n - CHUNK_SAMPLES) // 2
        chunk = audio[s:s + CHUNK_SAMPLES]
    else:
        pad = CHUNK_SAMPLES - n
        chunk = np.pad(audio, (pad // 2, pad - pad // 2))
    return chunk.astype(np.float32)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Class order must match the server's test set → read it from the cache.
    classes = [str(c) for c in np.load(CACHE, allow_pickle=True)["classes"]]
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    # Build local dataset once: this node's clips → spectrograms (STFT on edge).
    manifest = build_manifest(args.node_id)[:args.max_samples]
    stft = STFTProcessor(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    print(f"[client-{args.node_id}] building {len(manifest)} local spectrograms...",
          flush=True)
    X_local, y_local = [], []
    for row in manifest:
        wav = AUDIO_ROOT / f"fold{row['fold']}" / row["slice_file_name"]
        X_local.append(stft.process(load_chunk(wav)).astype(np.float32))
        y_local.append(cls_to_idx[row["class"]])
    X_local = np.stack(X_local)[..., np.newaxis]
    y_local = np.asarray(y_local, dtype=np.int32)
    print(f"[client-{args.node_id}] local set: X={X_local.shape}", flush=True)

    model = AudioCNN(num_classes=len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    inbox: queue.Queue = queue.Queue()
    mqtt_client = MQTTClient(broker=args.broker, port=args.port, node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"[client-{args.node_id}] ERROR: cannot connect to broker",
              file=sys.stderr)
        return 1
    mqtt_client.subscribe(GLOBAL_TOPIC, lambda t, p: inbox.put(p), qos=1)

    weights_topic = f"thesis/edge/{args.node_id}/weights"
    bytes_uploaded = 0
    round_times: list[dict] = []
    t_start = time.perf_counter()

    print(f"[client-{args.node_id}] waiting for global model...", flush=True)
    while True:
        try:
            payload = inbox.get(timeout=120.0)
        except queue.Empty:
            print(f"[client-{args.node_id}] no global model in 120s — exiting",
                  file=sys.stderr)
            break
        if not payload:  # cleared retained message at shutdown
            continue
        global_weights, meta = deserialize_weights(payload)
        if meta.get("done"):
            print(f"[client-{args.node_id}] server signaled done", flush=True)
            break

        rnd = meta["round"]
        model.set_weights(global_weights)
        # Fresh local optimizer each round: standard FedAvg does not carry
        # optimizer state across rounds. Reusing Adam's momentum after
        # set_weights() resets the weights corrupts the first steps.
        # LR comes from the server (per-round), so the harness can sweep it.
        local_lr = meta.get("local_lr", LR)
        model.compile(optimizer=tf.keras.optimizers.Adam(local_lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.perf_counter()
        model.fit(X_local, y_local, epochs=meta["local_epochs"],
                  batch_size=32, verbose=0)
        train_time = time.perf_counter() - t0

        payload_out = serialize_weights(model.get_weights(), {
            "node_id": args.node_id, "round": rnd, "n_samples": len(y_local)})
        if not mqtt_client.publish(weights_topic, payload_out, qos=1):
            print(f"[client-{args.node_id}] publish failed round {rnd}",
                  file=sys.stderr)
        bytes_uploaded += len(payload_out)
        round_times.append({"round": rnd, "local_train_seconds": round(train_time, 2)})
        print(f"[client-{args.node_id}] round {rnd}: trained {meta['local_epochs']} "
              f"epoch(s) in {train_time:.1f}s, sent {len(payload_out) / 1024:.0f} KB",
              flush=True)

    time.sleep(1.0)
    mqtt_client.disconnect()

    wall = time.perf_counter() - t_start
    result = {
        "node_id": args.node_id,
        "strategy": "federated",
        "local_samples": int(len(y_local)),
        "wall_seconds": round(wall, 2),
        "rounds_trained": len(round_times),
        "bytes_uploaded": bytes_uploaded,
        "round_times": round_times,
        "total_local_train_seconds": round(
            sum(r["local_train_seconds"] for r in round_times), 2),
    }
    out = run_dir / f"client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[client-{args.node_id}] done: {len(round_times)} rounds, "
          f"{bytes_uploaded / 1024 / 1024:.2f} MB uploaded, "
          f"{result['total_local_train_seconds']:.1f}s local training → {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
