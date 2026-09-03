"""
Strategy C Pi Client — federated, trains on the Pi, ships only weights
======================================================================

The real-hardware counterpart of strategy_c_client.py. The privacy claim
of the whole thesis lives here: audio is captured, STFT'd and trained on
this device, and the only thing that ever crosses the network is a weight
vector.

    receive global model (round r) -> set_weights -> fit() locally on the
    enrolled samples -> publish weights + n_samples -> wait for round r+1

Contrast with the other two live clients, which all train on the SAME
enrolled data:

    A  ships raw audio          -> reconstructible speech leaves the Pi
    B  ships mel-spectrograms   -> lossy, but still derived from the audio
    C  ships weights only       -> the audio never leaves the device

Local training on a Pi 5 measures ~3.8 s per epoch over 200 samples
(110k-parameter AudioCNN, batch 8), so a federated round completes in
seconds — this is what "model freshness in minutes, not hours" means in
practice, and the round times land in the report to prove it.

Run on the Pi (start strategy_c_server.py on the laptop first):

    .venv/bin/python -m src.experiments.strategy_c_pi_client \
        --node-id A --broker <LAPTOP_LAN_IP> \
        --dataset results/live/live_dataset.npz

Writes {--run-dir}/pi_client_{node_id}.json with per-round LOCAL train time
and upload bytes.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.experiments.pi_live import build_live_model, load_dataset  # noqa: E402

GLOBAL_TOPIC = "thesis/server/model/global"
LR = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True, help="Edge node id (A or B)")
    p.add_argument("--broker", required=True, help="Laptop IP running mosquitto")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--dataset", default="results/live/live_dataset.npz",
                   help="Enrolled dataset from pi_enroll.py")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Small batches suit the Pi's memory and give more "
                        "gradient steps from a small enrolled set")
    p.add_argument("--shard", default=None,
                   help="'i/n' — train on only this node's slice, so two Pis "
                        "hold disjoint data (the non-IID story)")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="Give up if no global model arrives in N seconds")
    p.add_argument("--run-dir", default="results/live/strategy_c")
    return p.parse_args()


def serialize_weights(weights: list, meta: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=json.dumps(meta),
                        **{f"w{i}": w for i, w in enumerate(weights)})
    return buf.getvalue()


def deserialize_weights(raw: bytes):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    n = sum(1 for k in z.files if k.startswith("w"))
    return [z[f"w{i}"] for i in range(n)], meta


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(Path(args.dataset))
    audio, y, classes = ds["audio_train"], ds["y_train"], ds["classes"]

    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        audio, y = audio[i::n], y[i::n]
        print(f"[pi-{args.node_id}] shard {args.shard}: {len(audio)} samples",
              flush=True)

    # STFT on the edge — same parameters as B and the servers, so the
    # feature space matches across every strategy.
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    print(f"[pi-{args.node_id}] building {len(audio)} local spectrograms...",
          flush=True)
    t0 = time.perf_counter()
    X_local = stft.process_batch(audio)[..., np.newaxis].astype(np.float32)
    stft_seconds = time.perf_counter() - t0
    y_local = np.asarray(y, dtype=np.int32)
    print(f"[pi-{args.node_id}] local set: X={X_local.shape} "
          f"({stft_seconds:.1f}s edge STFT), classes={classes}", flush=True)

    model, warm_started = build_live_model(len(classes), learning_rate=LR)

    inbox: queue.Queue = queue.Queue()
    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"ERROR: cannot reach broker {args.broker}:{args.port}",
              file=sys.stderr)
        return 1
    mqtt_client.subscribe(GLOBAL_TOPIC, lambda t, p: inbox.put(p), qos=1)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    weights_topic = f"thesis/edge/{args.node_id}/weights"
    bytes_uploaded = 0
    round_times: list[dict] = []
    t_start = time.perf_counter()

    print(f"[pi-{args.node_id}] waiting for global model...", flush=True)
    while not stop["flag"]:
        try:
            payload = inbox.get(timeout=args.timeout)
        except queue.Empty:
            print(f"[pi-{args.node_id}] no global model in {args.timeout:.0f}s "
                  f"— exiting", file=sys.stderr)
            break
        if not payload:  # cleared retained message at shutdown
            continue

        global_weights, meta = deserialize_weights(payload)
        if meta.get("done"):
            print(f"[pi-{args.node_id}] server signaled done", flush=True)
            break

        rnd = meta["round"]
        model.set_weights(global_weights)
        # Fresh optimizer each round: FedAvg does not carry optimizer state
        # across rounds, and reusing Adam's momentum after set_weights()
        # corrupts the first steps.
        local_lr = meta.get("local_lr", LR)
        model.compile(optimizer=tf.keras.optimizers.Adam(local_lr),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        t0 = time.perf_counter()
        model.fit(X_local, y_local, epochs=meta["local_epochs"],
                  batch_size=args.batch_size, verbose=0)
        train_time = time.perf_counter() - t0

        payload_out = serialize_weights(model.get_weights(), {
            "node_id": args.node_id, "round": rnd, "n_samples": len(y_local)})
        if not mqtt_client.publish(weights_topic, payload_out, qos=1):
            print(f"[pi-{args.node_id}] publish failed round {rnd}",
                  file=sys.stderr)
        bytes_uploaded += len(payload_out)
        round_times.append({"round": rnd,
                            "local_train_seconds": round(train_time, 2)})
        print(f"[pi-{args.node_id}] round {rnd}: trained "
              f"{meta['local_epochs']} epoch(s) on {len(y_local)} samples in "
              f"{train_time:.1f}s, sent {len(payload_out) / 1024:.0f} KB",
              flush=True)

    time.sleep(1.0)
    mqtt_client.disconnect()

    result = {
        "node_id": args.node_id,
        "strategy": "C_federated",
        "mode": "live_vocabulary",
        "dataset": str(args.dataset),
        "classes": classes,
        "warm_started": warm_started,
        "local_samples": int(len(y_local)),
        "edge_stft_seconds": round(stft_seconds, 2),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "rounds_trained": len(round_times),
        "bytes_uploaded": bytes_uploaded,
        "round_times": round_times,
        "total_local_train_seconds": round(
            sum(r["local_train_seconds"] for r in round_times), 2),
        "raw_audio_bytes_never_sent": int(audio.nbytes),
    }
    out = run_dir / f"pi_client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[pi-{args.node_id}] done: {len(round_times)} rounds, "
          f"{bytes_uploaded / 1024:.0f} KB uploaded, "
          f"{result['total_local_train_seconds']:.1f}s local training -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
