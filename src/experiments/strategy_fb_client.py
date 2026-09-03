"""
Strategy FB Client — Hybrid with the FUSION model (F6)
======================================================

The fusion-era version of Strategy B. The edge computes BOTH compact
representations and ships them — never the frame, never raw audio:

    per sample:  spectrogram (51x128 float16)  ~13 KB
               + vision feature vector (576 float16)  ~1.1 KB
               + label

Simulation source: the VGGSound paired cache (spectrograms) + fv_cache
(precomputed backbone outputs = what the edge's frozen MobileNetV3 would
emit per capture). Clients A/B stream disjoint clip-halves of the train
split.

Spawned by scripts/run_strategy.py --strategy fb.
Output: {run-dir}/client_{node_id}.json with upload metrics.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
FV_CACHE = REPO_ROOT / "data/vggsound/fv_cache.npz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=100000,
                   help="Cap (default: this node's full half)")
    p.add_argument("--stream-rate-ms", type=int, default=50,
                   help="Per-sample cadence (kept small: bandwidth per sample "
                        "is the metric, wall time is not)")
    p.add_argument("--buffer-size", type=int, default=250)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def node_samples(node_id: str):
    """Disjoint clip-split of the train set: A = even clips, B = odd."""
    d = np.load(CACHE, allow_pickle=True)
    fv = np.load(FV_CACHE, allow_pickle=True)["fv"]
    idx, X_aud, y = d["img_idx_train"], d["X_aud_train"], d["y_train"]
    clips = sorted(set(idx.tolist()))
    offset = {"A": 0, "B": 1}.get(node_id)
    if offset is None:
        raise SystemExit(f"unsupported --node-id {node_id!r}")
    mine = set(clips[offset::2])
    mask = np.array([i in mine for i in idx])
    classes = [str(c) for c in d["classes"]]
    return (X_aud[mask], fv[idx[mask]], y[mask], classes)


def serialize_batch(specs, fvs, labels) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        spec=specs.astype(np.float16),
        fv=fvs.astype(np.float16),
        labels=np.asarray(labels, dtype=np.int32),
    )
    return zlib.compress(buf.getvalue())


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    X_aud, X_fv, y, classes = node_samples(args.node_id)
    n = min(len(y), args.max_samples)
    print(f"[client-{args.node_id}] streaming {n} paired samples "
          f"(spec + fv + label)", flush=True)

    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"[client-{args.node_id}] ERROR: broker unreachable",
              file=sys.stderr)
        return 1
    topic = f"thesis/edge/{args.node_id}/data"

    rate_s = args.stream_rate_ms / 1000.0
    bytes_uploaded = 0
    batches = 0
    buf_specs, buf_fvs, buf_labels = [], [], []
    t_start = time.perf_counter()

    def flush():
        nonlocal bytes_uploaded, batches
        payload = serialize_batch(np.stack(buf_specs)[..., 0],
                                  np.stack(buf_fvs), buf_labels)
        if mqtt_client.publish(topic, payload, qos=1):
            bytes_uploaded += len(payload)
            batches += 1
            print(f"[client-{args.node_id}] batch {batches} uploaded "
                  f"({len(payload) / 1024:.0f} KB, "
                  f"{bytes_uploaded / 1024 / 1024:.2f} MB total)", flush=True)
        buf_specs.clear(); buf_fvs.clear(); buf_labels.clear()

    for i in range(n):
        buf_specs.append(X_aud[i])
        buf_fvs.append(X_fv[i])
        buf_labels.append(int(y[i]))
        if len(buf_labels) >= args.buffer_size:
            flush()
        next_tick = t_start + (i + 1) * rate_s
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    if buf_labels:
        print(f"[client-{args.node_id}] flushing final partial batch "
              f"({len(buf_labels)})", flush=True)
        flush()

    time.sleep(2.0)
    mqtt_client.disconnect()

    wall = time.perf_counter() - t_start
    result = {
        "node_id": args.node_id,
        "strategy": "fusion_hybrid",
        "samples_streamed": n,
        "wall_seconds": round(wall, 2),
        "batches_uploaded": batches,
        "bytes_uploaded": bytes_uploaded,
        "bytes_per_sample": round(bytes_uploaded / max(n, 1), 1),
    }
    out = run_dir / f"client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[client-{args.node_id}] done: {n} samples, "
          f"{bytes_uploaded / 1024 / 1024:.2f} MB "
          f"({result['bytes_per_sample']:.0f} B/sample) → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
