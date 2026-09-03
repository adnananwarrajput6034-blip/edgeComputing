"""
Strategy FB Server — Hybrid with the FUSION model (F6)
======================================================

Receives (spectrogram + vision FV + label) batches, pools them, and trains
the FVFusionModel — warm-started from models/fv_fusion.keras (the verified
FV-equivalent of the proven fusion model). Each round it evaluates BOTH:

    clean accuracy     — real FVs + audio
    blackout accuracy  — black-frame FV + audio (the fog condition)

so the run shows whether continued strategy training preserves the graceful
degradation property, round by round.

Same CLI/server.json contract as the other servers (harness-compatible);
rounds[] entries additionally carry blackout_accuracy.
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
import zlib
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
FV_CACHE = REPO_ROOT / "data/vggsound/fv_cache.npz"
WARM_START = REPO_ROOT / "models/fv_fusion.keras"
DATA_TOPIC = "thesis/edge/+/data"
MODEL_TOPIC = "thesis/server/model/global"
READY_MARKER = "Server ready. Waiting for edge batches"
BATCH_SIZE = 32
LR = 1e-4  # gentle: continuing training on a converged fusion model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--num-rounds", type=int, default=6)
    p.add_argument("--train-trigger-samples", type=int, default=500)
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


class StrategyFBServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        d = np.load(CACHE, allow_pickle=True)
        fvz = np.load(FV_CACHE, allow_pickle=True)
        fv, self.fv_black = fvz["fv"], fvz["fv_black"]
        self.classes = [str(c) for c in d["classes"]]
        idx = d["img_idx_test"]
        self.Xa_test, self.y_test = d["X_aud_test"], d["y_test"]
        self.fv_test = fv[idx]
        self.fv_test_blk = np.tile(self.fv_black, (len(self.y_test), 1))

        print(f"Warm-starting FVFusionModel from {WARM_START}", flush=True)
        self.model = tf.keras.models.load_model(WARM_START)
        if getattr(self.model, "optimizer", None) is None:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                               loss="sparse_categorical_crossentropy",
                               metrics=["accuracy"])

        self.inbox: queue.Queue = queue.Queue()
        self.stop_requested = False
        self.spec_pool, self.fv_pool, self.y_pool = [], [], []
        self.pool_count = 0
        self.per_client_bytes: dict[str, int] = {}
        self.per_client_samples: dict[str, int] = {}
        self.rounds: list[dict] = []
        self.bytes_broadcast_total = 0

        self.mqtt = MQTTClient(broker=args.broker, port=args.port,
                               node_id="server")

    def connect(self) -> bool:
        if not self.mqtt.connect():
            return False
        return self.mqtt.subscribe(DATA_TOPIC,
                                   lambda t, p: self.inbox.put((t, p)), qos=1)

    def ingest(self, topic: str, payload: bytes) -> None:
        node_id = topic.split("/")[2]
        self.per_client_bytes[node_id] = (
            self.per_client_bytes.get(node_id, 0) + len(payload))
        z = np.load(io.BytesIO(zlib.decompress(payload)), allow_pickle=False)
        spec = z["spec"].astype(np.float32)[..., np.newaxis]
        fv = z["fv"].astype(np.float32)
        labels = z["labels"]
        self.spec_pool.append(spec)
        self.fv_pool.append(fv)
        self.y_pool.append(labels)
        self.pool_count += len(labels)
        self.per_client_samples[node_id] = (
            self.per_client_samples.get(node_id, 0) + len(labels))
        print(f"Batch from {node_id}: {len(labels)} paired samples "
              f"({len(payload) / 1024:.0f} KB) — pool={self.pool_count}",
              flush=True)

    def evaluate(self) -> tuple[float, float]:
        p = self.model.predict({"vision_fv_input": self.fv_test,
                                "audio_input": self.Xa_test}, verbose=0)
        clean = float((p.argmax(1) == self.y_test).mean())
        p = self.model.predict({"vision_fv_input": self.fv_test_blk,
                                "audio_input": self.Xa_test}, verbose=0)
        blackout = float((p.argmax(1) == self.y_test).mean())
        return clean, blackout

    def maybe_train(self) -> None:
        while (len(self.rounds) < self.args.num_rounds
               and self.pool_count >= (len(self.rounds) + 1)
               * self.args.train_trigger_samples):
            self.run_round()

    def run_round(self) -> None:
        rnd = len(self.rounds) + 1
        X = {"vision_fv_input": np.concatenate(self.fv_pool),
             "audio_input": np.concatenate(self.spec_pool)}
        y = np.concatenate(self.y_pool)
        print(f"\n=== Round {rnd}/{self.args.num_rounds}: "
              f"training on {len(y)} pooled samples ===", flush=True)
        t0 = time.perf_counter()
        self.model.fit(X, y, epochs=self.args.epochs_per_round,
                       batch_size=BATCH_SIZE, verbose=2)
        train_time = time.perf_counter() - t0
        clean, blackout = self.evaluate()

        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"w{i}": w for i, w in
                                    enumerate(self.model.get_weights())})
        payload = buf.getvalue()
        self.mqtt.publish(MODEL_TOPIC, payload, qos=1)
        self.bytes_broadcast_total += len(payload)

        self.rounds.append({
            "round": rnd, "buffer_samples": int(len(y)),
            "test_accuracy": round(clean, 4),
            "blackout_accuracy": round(blackout, 4),
            "train_time_seconds": round(train_time, 2),
            "broadcast_bytes": len(payload),
        })
        print(f"=== Round {rnd}: clean={clean:.3f} blackout={blackout:.3f} "
              f"train={train_time:.1f}s ===\n", flush=True)
        self.write_report()

    def write_report(self) -> None:
        (self.run_dir / "server.json").write_text(json.dumps({
            "strategy": "fusion_hybrid",
            "num_rounds_target": self.args.num_rounds,
            "train_trigger_samples": self.args.train_trigger_samples,
            "epochs_per_round": self.args.epochs_per_round,
            "warm_started": True,
            "classes": self.classes,
            "rounds": self.rounds,
            "samples_received_total": self.pool_count,
            "per_client_samples": self.per_client_samples,
            "per_client_bytes_total": self.per_client_bytes,
            "bytes_broadcast_total": self.bytes_broadcast_total,
        }, indent=2))

    def run(self) -> int:
        if not self.connect():
            print("ERROR: broker unreachable", file=sys.stderr)
            return 1
        clean, blackout = self.evaluate()
        print(f"Baseline before rounds: clean={clean:.3f} "
              f"blackout={blackout:.3f}", flush=True)
        print(READY_MARKER, flush=True)

        while not self.stop_requested:
            try:
                topic, payload = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            self.ingest(topic, payload)
            self.maybe_train()

        while True:
            try:
                topic, payload = self.inbox.get_nowait()
            except queue.Empty:
                break
            self.ingest(topic, payload)
        self.write_report()
        self.mqtt.disconnect()
        print(f"Server stopped. {len(self.rounds)}/{self.args.num_rounds} "
              f"rounds, {self.pool_count} samples.", flush=True)
        return 0


def main() -> int:
    args = parse_args()
    server = StrategyFBServer(args)

    def handle_term(signum, frame):
        print(f"Received signal {signum} — shutting down", flush=True)
        server.stop_requested = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
