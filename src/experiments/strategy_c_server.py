"""
Strategy C Server — Federated (FedAvg over MQTT, hand-rolled)
============================================================

The server does NO training — it only averages weights. Round-synchronized
federated learning over the same MQTT broker as A/B (so bandwidth is
measured on the same transport → comparable):

    broadcast global model (retained) → wait for BOTH clients' weights →
    FedAvg aggregate → evaluate on fold-10 test set → broadcast next global

Contrast with A/B: there the server ran model.fit on uploaded data. Here
the clients train; the server just runs FedAvgAggregator.aggregate(). That
near-zero server compute is Strategy C's defining trait.

Design notes:
  - Global broadcasts are RETAINED so a client subscribing late still gets
    the current round's model (kills the startup race without a handshake).
  - Messages carry a round number; stale weights from a prior round are
    dropped. Quorum = both clients before aggregating.
  - The server tells clients how many local epochs to run (from
    --epochs-per-round), so the generic harness controls C unchanged.

Same server.json schema as A/B (rounds[], per_client_bytes_total, etc.);
train_time_seconds here is aggregation+eval time (small by design), while
the real compute cost lives in client_*.json.
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
from src.server.training.models.audio_cnn import AudioCNN  # noqa: E402

CACHE = REPO_ROOT / "data/urbansound8k/cache.npz"
WARM_START = REPO_ROOT / "models/audio_cnn_urbansound8k.keras"
WEIGHTS_TOPIC = "thesis/edge/+/weights"
GLOBAL_TOPIC = "thesis/server/model/global"
READY_MARKER = "Server ready. Waiting for edge batches"
N_CLIENTS = 2
LR = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--train-trigger-samples", type=int, default=1000,
                   help="unused for C (federated is round-driven); accepted "
                        "for harness compatibility")
    p.add_argument("--epochs-per-round", type=int, default=1,
                   help="LOCAL epochs each client trains per round")
    p.add_argument("--local-lr", type=float, default=1e-3,
                   help="LOCAL learning rate clients use (sent to clients "
                        "each round; lower stabilizes FedAvg on a warm-started model)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--num-clients", type=int, default=N_CLIENTS,
                   help="Clients that must report before FedAvg aggregates a "
                        "round. Defaults to 2 (the science runs); set 1 when "
                        "demoing with a single Pi, or the server waits "
                        "forever for a second node that never arrives.")
    p.add_argument("--save-model", default=None,
                   help="Write the trained model here (.keras) plus a "
                        "<name>.classes.json sidecar, so pi_infer.py can "
                        "load it for live inference. Without this the "
                        "trained model is discarded at shutdown.")
    p.add_argument("--live-dataset", default=None,
                   help="Enrolled dataset from pi_enroll.py. Replaces the "
                        "UrbanSound8K vocabulary and test set with the "
                        "classes captured live, and resizes the global model "
                        "head to match the Pi clients.")
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
    weights = [z[f"w{i}"] for i in range(n)]
    return weights, meta


class StrategyCServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if args.live_dataset:
            # Live track: the global model must have the same head width as
            # the Pi clients' local models, or set_weights() on the client
            # fails. Both sides build it from the enrolled class list.
            from src.edge.processing.stft import STFTProcessor
            from src.experiments.pi_live import build_live_model, load_dataset
            print(f"Live dataset: {args.live_dataset}", flush=True)
            ds = load_dataset(Path(args.live_dataset))
            self.classes = ds["classes"]
            eval_stft = STFTProcessor(sample_rate=16000, n_fft=512,
                                      hop_length=160, n_mels=128,
                                      window_length=400, normalize=True)
            self.X_test = eval_stft.process_batch(
                ds["audio_test"])[..., np.newaxis].astype(np.float32)
            self.y_test = ds["y_test"]
            self.model, self.warm_started = build_live_model(len(self.classes))
            print(f"Live vocabulary {self.classes}, "
                  f"test set {self.X_test.shape}", flush=True)
        else:
            print(f"Loading eval cache: {CACHE}", flush=True)
            data = np.load(CACHE, allow_pickle=True)
            self.X_test, self.y_test = data["X_test"], data["y_test"]
            self.classes = [str(c) for c in data["classes"]]

            if WARM_START.exists():
                print(f"Warm-starting global model from {WARM_START}", flush=True)
                self.model = tf.keras.models.load_model(WARM_START)
                self.warm_started = True
            else:
                print("No pre-trained weights — global model from scratch",
                      flush=True)
                self.model = AudioCNN(num_classes=len(self.classes))
                self.warm_started = False
        if getattr(self.model, "optimizer", None) is None:
            self.model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                               loss="sparse_categorical_crossentropy",
                               metrics=["accuracy"])

        self.aggregator = FedAvgAggregator(initial_weights=self.model.get_weights())

        self.inbox: queue.Queue = queue.Queue()
        self.stop_requested = False
        self.current_round = 1
        self.pending: dict[str, tuple] = {}   # node_id -> (weights, n_samples)
        self.per_client_bytes: dict[str, int] = {}
        self.per_client_samples: dict[str, int] = {}
        self.rounds: list[dict] = []
        self.bytes_broadcast_total = 0

        self.mqtt = MQTTClient(broker=args.broker, port=args.port, node_id="server")

    def _on_weights(self, topic: str, payload: bytes) -> None:
        self.inbox.put((topic, payload))

    def connect(self) -> bool:
        if not self.mqtt.connect():
            return False
        return self.mqtt.subscribe(WEIGHTS_TOPIC, self._on_weights, qos=1)

    def broadcast_global(self, round_no: int, done: bool) -> None:
        payload = serialize_weights(self.model.get_weights(), {
            "round": round_no,
            "local_epochs": self.args.epochs_per_round,
            "local_lr": self.args.local_lr,
            "done": done,
        })
        self.mqtt.publish(GLOBAL_TOPIC, payload, qos=1, retain=True)
        self.bytes_broadcast_total += len(payload)

    def handle_weights(self, topic: str, payload: bytes) -> None:
        node_id = topic.split("/")[2]
        weights, meta = deserialize_weights(payload)
        if meta.get("round") != self.current_round:
            return  # stale weights from a prior round — ignore
        self.per_client_bytes[node_id] = (
            self.per_client_bytes.get(node_id, 0) + len(payload))
        self.per_client_samples[node_id] = meta["n_samples"]
        self.pending[node_id] = (weights, meta["n_samples"])
        print(f"Round {self.current_round}: weights from {node_id} "
              f"({meta['n_samples']} samples, {len(payload) / 1024:.0f} KB)",
              flush=True)

    def aggregate_round(self) -> None:
        client_weights = [w for w, _ in self.pending.values()]
        sample_counts = [n for _, n in self.pending.values()]
        t0 = time.perf_counter()
        new_global = self.aggregator.aggregate(client_weights, sample_counts)
        self.model.set_weights(new_global)
        _, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        agg_time = time.perf_counter() - t0

        self.rounds.append({
            "round": self.current_round,
            "buffer_samples": int(sum(sample_counts)),
            "test_accuracy": float(test_acc),
            "train_time_seconds": round(agg_time, 2),  # aggregation+eval (server does no training)
            "n_clients": len(self.pending),
        })
        print(f"=== Round {self.current_round} FedAvg: test_acc={test_acc:.3f}, "
              f"agg={agg_time:.2f}s, {len(self.pending)} clients ===", flush=True)
        self.write_report()

        self.current_round += 1
        self.pending = {}

    def write_report(self) -> None:
        report = {
            "strategy": "federated",
            "num_rounds_target": self.args.num_rounds,
            "epochs_per_round": self.args.epochs_per_round,
            "warm_started": self.warm_started,
            "classes": self.classes,
            "rounds": self.rounds,
            "per_client_samples": self.per_client_samples,
            "per_client_bytes_total": self.per_client_bytes,
            "bytes_broadcast_total": self.bytes_broadcast_total,
        }
        (self.run_dir / "server.json").write_text(json.dumps(report, indent=2))

    def save_model(self) -> None:
        """Persist the trained model so it can actually be USED afterwards.

        Without this the demo trains a classifier and then discards it —
        weights are broadcast over MQTT and never land on disk. pi_infer.py
        loads what this writes. The class list rides alongside in a sidecar
        json, because a .keras file records the head width but not what the
        outputs mean.
        """
        if not self.args.save_model:
            return
        out = Path(self.args.save_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(out)
        out.with_suffix(".classes.json").write_text(
            json.dumps({"classes": self.classes}, indent=2))
        print(f"Saved trained model -> {out} "
              f"({len(self.classes)} classes: {self.classes})", flush=True)

    def run(self) -> int:
        if not self.connect():
            print(f"ERROR: cannot connect/subscribe to broker "
                  f"{self.args.broker}:{self.args.port}", file=sys.stderr)
            return 1

        _, base_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Baseline test accuracy before any round: {base_acc:.3f}", flush=True)

        # Kick off round 1 (retained, so late-subscribing clients still get it).
        self.broadcast_global(round_no=1, done=False)
        print(READY_MARKER, flush=True)

        while not self.stop_requested and len(self.rounds) < self.args.num_rounds:
            try:
                topic, payload = self.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle_weights(topic, payload)
            if len(self.pending) >= self.args.num_clients:
                self.aggregate_round()
                done = len(self.rounds) >= self.args.num_rounds
                self.broadcast_global(
                    round_no=self.current_round if not done else self.args.num_rounds,
                    done=done)

        # Ensure clients get the stop signal even on early termination.
        self.broadcast_global(round_no=self.args.num_rounds, done=True)
        self.write_report()
        # Clear the retained global so a later experiment doesn't inherit it.
        self.mqtt.publish(GLOBAL_TOPIC, b"", qos=1, retain=True)
        self.save_model()
        self.mqtt.disconnect()
        print(f"Server stopped. {len(self.rounds)}/{self.args.num_rounds} rounds.",
              flush=True)
        return 0


def main() -> int:
    args = parse_args()
    server = StrategyCServer(args)

    def handle_term(signum, frame):
        print(f"Received signal {signum} — shutting down", flush=True)
        server.stop_requested = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
