"""
Strategy FC Client — Federated with the FUSION model (F6)
=========================================================

The fusion-era Strategy C: each client holds its own paired data (spectrogram
+ vision FV + label — clip-disjoint half of the train split), receives the
global FVFusionModel each round, trains locally, ships only weights. Frames
AND audio never leave the device.

Same round protocol as strategy_c_client.py (retained global broadcast,
round tags, fresh local optimizer per round, done flag).

Spawned by scripts/run_strategy.py --strategy fc.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.server.training.models.fusion_model import FVFusionModel  # noqa: E402

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
FV_CACHE = REPO_ROOT / "data/vggsound/fv_cache.npz"
GLOBAL_TOPIC = "thesis/server/model/global"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=100000)
    p.add_argument("--stream-rate-ms", type=int, default=0,
                   help="harness compatibility (unused)")
    p.add_argument("--buffer-size", type=int, default=0,
                   help="harness compatibility (unused)")
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def serialize_weights(weights, meta) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=json.dumps(meta),
                        **{f"w{i}": w for i, w in enumerate(weights)})
    return buf.getvalue()


def deserialize_weights(raw: bytes):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = sum(1 for k in z.files if k.startswith("w"))
    return [z[f"w{i}"] for i in range(n)], meta


def node_samples(node_id: str):
    d = np.load(CACHE, allow_pickle=True)
    fv = np.load(FV_CACHE, allow_pickle=True)["fv"]
    idx, X_aud, y = d["img_idx_train"], d["X_aud_train"], d["y_train"]
    clips = sorted(set(idx.tolist()))
    offset = {"A": 0, "B": 1}.get(node_id)
    if offset is None:
        raise SystemExit(f"unsupported --node-id {node_id!r}")
    mine = set(clips[offset::2])
    mask = np.array([i in mine for i in idx])
    n_cls = len(d["classes"])
    return X_aud[mask], fv[idx[mask]], y[mask], n_cls


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    Xa, Xf, y, n_cls = node_samples(args.node_id)
    n = min(len(y), args.max_samples)
    Xa, Xf, y = Xa[:n], Xf[:n], y[:n]
    X_local = {"vision_fv_input": Xf, "audio_input": Xa}
    print(f"[client-{args.node_id}] local paired set: {n} samples", flush=True)

    model = FVFusionModel(num_classes=n_cls)

    inbox: queue.Queue = queue.Queue()
    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"[client-{args.node_id}] ERROR: broker unreachable",
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
        if not payload:
            continue
        global_weights, meta = deserialize_weights(payload)
        if meta.get("done"):
            print(f"[client-{args.node_id}] server signaled done", flush=True)
            break

        rnd = meta["round"]
        model.set_weights(global_weights)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(meta.get("local_lr", 1e-4)),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.perf_counter()
        model.fit(X_local, y, epochs=meta.get("local_epochs", 1),
                  batch_size=32, verbose=0)
        train_time = time.perf_counter() - t0

        out = serialize_weights(model.get_weights(), {
            "node_id": args.node_id, "round": rnd, "n_samples": int(n)})
        if not mqtt_client.publish(weights_topic, out, qos=1):
            print(f"[client-{args.node_id}] publish failed round {rnd}",
                  file=sys.stderr)
        bytes_uploaded += len(out)
        round_times.append({"round": rnd,
                            "local_train_seconds": round(train_time, 2)})
        print(f"[client-{args.node_id}] round {rnd}: trained in "
              f"{train_time:.1f}s, sent {len(out) / 1024:.0f} KB", flush=True)

    time.sleep(1.0)
    mqtt_client.disconnect()

    result = {
        "node_id": args.node_id,
        "strategy": "fusion_federated",
        "local_samples": int(n),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "rounds_trained": len(round_times),
        "bytes_uploaded": bytes_uploaded,
        "round_times": round_times,
        "total_local_train_seconds": round(
            sum(r["local_train_seconds"] for r in round_times), 2),
    }
    out_path = run_dir / f"client_{args.node_id}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[client-{args.node_id}] done: {len(round_times)} rounds, "
          f"{bytes_uploaded / 1024 / 1024:.2f} MB uploaded → {out_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
