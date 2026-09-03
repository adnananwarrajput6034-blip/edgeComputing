"""
Strategy FC Server — Federated with the FUSION model (F6)
=========================================================

FedAvg over FVFusionModel weights. Same round protocol as
strategy_c_server.py (retained global broadcast with round/local_epochs/
local_lr, quorum of both clients, stale-round rejection). Warm-starts the
global model from models/fv_fusion.keras. Evaluates clean AND blackout
accuracy each round — showing whether federated training preserves the
graceful-degradation property.

Default local LR is 1e-4 (gentler than 1e-3; the C experiments showed
converged models need small local steps under FedAvg).
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
from src.server.aggregation.fedavg import FedAvgAggregator  # noqa: E402

CACHE = REPO_ROOT / "data/vggsound/paired_cache.npz"
FV_CACHE = REPO_ROOT / "data/vggsound/fv_cache.npz"
WARM_START = REPO_ROOT / "models/fv_fusion.keras"
WEIGHTS_TOPIC = "thesis/edge/+/weights"
GLOBAL_TOPIC = "thesis/server/model/global"
READY_MARKER = "Server ready. Waiting for edge batches"
N_CLIENTS = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--train-trigger-samples", type=int, default=0,
                   help="harness compatibility (unused — round-driven)")
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--local-lr", type=float, default=1e-4)
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


class StrategyFCServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        d = np.load(CACHE, allow_pickle=True)
        fvz = np.load(FV_CACHE, allow_pickle=True)
        fv, fv_black = fvz["fv"], fvz["fv_black"]
        self.classes = [str(c) for c in d["classes"]]
        idx = d["img_idx_test"]
        self.Xa_test, self.y_test = d["X_aud_test"], d["y_test"]
        self.fv_test = fv[idx]
        self.fv_test_blk = np.tile(fv_black, (len(self.y_test), 1))

        print(f"Warm-starting global FVFusionModel from {WARM_START}",
              flush=True)
        self.model = tf.keras.models.load_model(WARM_START)
        self.aggregator = FedAvgAggregator(
            initial_weights=self.model.get_weights())

        self.inbox: queue.Queue = queue.Queue()
        self.stop_requested = False
        self.current_round = 1
        self.pending: dict[str, tuple] = {}
        self.per_client_bytes: dict[str, int] = {}
        self.per_client_samples: dict[str, int] = {}
        self.rounds: list[dict] = []
        self.bytes_broadcast_total = 0

        self.mqtt = MQTTClient(broker=args.broker, port=args.port,
                               node_id="server")

    def connect(self) -> bool:
        if not self.mqtt.connect():
            return False
        return self.mqtt.subscribe(WEIGHTS_TOPIC,
                                   lambda t, p: self.inbox.put((t, p)), qos=1)

    def broadcast_global(self, round_no: int, done: bool) -> None:
        payload = serialize_weights(self.model.get_weights(), {
            "round": round_no,
            "local_epochs": self.args.epochs_per_round,
            "local_lr": self.args.local_lr,
            "done": done,
        })
        self.mqtt.publish(GLOBAL_TOPIC, payload, qos=1, retain=True)
        self.bytes_broadcast_total += len(payload)

    def evaluate(self) -> tuple[float, float]:
        p = self.model.predict({"vision_fv_input": self.fv_test,
                                "audio_input": self.Xa_test}, verbose=0)
        clean = float((p.argmax(1) == self.y_test).mean())
        p = self.model.predict({"vision_fv_input": self.fv_test_blk,
                                "audio_input": self.Xa_test}, verbose=0)
        return clean, float((p.argmax(1) == self.y_test).mean())

    def handle_weights(self, topic: str, payload: bytes) -> None:
        node_id = topic.split("/")[2]
        weights, meta = deserialize_weights(payload)
        if meta.get("round") != self.current_round:
            return
        self.per_client_bytes[node_id] = (
            self.per_client_bytes.get(node_id, 0) + len(payload))
        self.per_client_samples[node_id] = meta["n_samples"]
        self.pending[node_id] = (weights, meta["n_samples"])
        print(f"Round {self.current_round}: weights from {node_id} "
              f"({len(payload) / 1024:.0f} KB)", flush=True)

    def aggregate_round(self) -> None:
        client_weights = [w for w, _ in self.pending.values()]
        counts = [c for _, c in self.pending.values()]
        t0 = time.perf_counter()
        self.model.set_weights(
            self.aggregator.aggregate(client_weights, counts))
        clean, blackout = self.evaluate()
        agg_time = time.perf_counter() - t0

        self.rounds.append({
            "round": self.current_round,
            "buffer_samples": int(sum(counts)),
            "test_accuracy": round(clean, 4),
            "blackout_accuracy": round(blackout, 4),
            "train_time_seconds": round(agg_time, 2),
        })
        print(f"=== Round {self.current_round} FedAvg: clean={clean:.3f} "
              f"blackout={blackout:.3f} agg={agg_time:.2f}s ===", flush=True)
        self.write_report()
        self.current_round += 1
        self.pending = {}

    def write_report(self) -> None:
        (self.run_dir / "server.json").write_text(json.dumps({
            "strategy": "fusion_federated",
            "num_rounds_target": self.args.num_rounds,
            "epochs_per_round": self.args.epochs_per_round,
            "local_lr": self.args.local_lr,
            "warm_started": True,
            "classes": self.classes,
            "rounds": self.rounds,
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
        self.broadcast_global(round_no=1, done=False)
        print(READY_MARKER, flush=True)

        while (not self.stop_requested
               and len(self.rounds) < self.args.num_rounds):
            try:
                topic, payload = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle_weights(topic, payload)
            if len(self.pending) >= N_CLIENTS:
                self.aggregate_round()
                done = len(self.rounds) >= self.args.num_rounds
                self.broadcast_global(
                    round_no=self.current_round if not done
                    else self.args.num_rounds, done=done)

        self.broadcast_global(round_no=self.args.num_rounds, done=True)
        self.write_report()
        self.mqtt.publish(GLOBAL_TOPIC, b"", qos=1, retain=True)
        self.mqtt.disconnect()
        print(f"Server stopped. {len(self.rounds)}/{self.args.num_rounds} "
              f"rounds.", flush=True)
        return 0


def main() -> int:
    args = parse_args()
    server = StrategyFCServer(args)

    def handle_term(signum, frame):
        print(f"Received signal {signum} — shutting down", flush=True)
        server.stop_requested = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
