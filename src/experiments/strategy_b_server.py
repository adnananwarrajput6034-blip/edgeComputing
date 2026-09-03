"""
Strategy B Server — Hybrid (spectrograms arrive ready-made)
===========================================================

Same as strategy_a_server.py EXCEPT the edge already did the STFT, so
ingest() just decompresses and pools — no server-side STFT. That moved
compute is Strategy B's defining trait (edge pays it, not the server).

    decompress → (already spectrograms) → pool →
    train when pool crosses --train-trigger-samples →
    evaluate on fold-10 test set → broadcast weights

Warm-starts from models/audio_cnn_urbansound8k.keras when present.

Spawned by scripts/run_strategy.py --strategy b. Same CLI contract and
server.json schema as the A server (the harness greps READY_MARKER and
parses rounds[]/bytes fields identically).
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
from src.server.training.models.audio_cnn import AudioCNN  # noqa: E402

CACHE = REPO_ROOT / "data/urbansound8k/cache.npz"
WARM_START = REPO_ROOT / "models/audio_cnn_urbansound8k.keras"
DATA_TOPIC = "thesis/edge/+/data"
MODEL_TOPIC = "thesis/server/model/global"
READY_MARKER = "Server ready. Waiting for edge batches"
BATCH_SIZE = 32
LR = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--num-rounds", type=int, default=10)
    p.add_argument("--train-trigger-samples", type=int, default=1000)
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help="Gradient batch size. The 32 default suits the "
                        "thousands of UrbanSound8K samples; a live enrolled "
                        "set of ~100 gives only 3 steps per epoch at that "
                        "size, so the model barely moves. Use 8 for live "
                        "runs.")
    p.add_argument("--save-model", default=None,
                   help="Write the trained model here (.keras) plus a "
                        "<name>.classes.json sidecar, so pi_infer.py can "
                        "load it for live inference. Without this the "
                        "trained model is discarded at shutdown.")
    p.add_argument("--live-dataset", default=None,
                   help="Enrolled dataset from pi_enroll.py. Replaces the "
                        "UrbanSound8K vocabulary and test set with the "
                        "classes captured live, and resizes the model head "
                        "to match.")
    return p.parse_args()


class StrategyBServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if args.live_dataset:
            # Live track: vocabulary and test set come from what the Pi
            # actually enrolled, not from UrbanSound8K. The test audio is
            # STFT'd here only to build the eval set — incoming batches
            # already arrive as spectrograms, which is Strategy B's point.
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
                print(f"Warm-starting from {WARM_START}", flush=True)
                self.model = tf.keras.models.load_model(WARM_START)
                self.warm_started = True
            else:
                print("No pre-trained weights found — training from scratch",
                      flush=True)
                self.model = AudioCNN(num_classes=len(self.classes))
                self.warm_started = False

        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        if getattr(self.model, "optimizer", None) is None:
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(LR),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

        self.inbox: queue.Queue = queue.Queue()
        self.stop_requested = False

        self.X_pool: list[np.ndarray] = []
        self.y_pool: list[np.ndarray] = []
        self.pool_count = 0
        self.per_client_bytes: dict[str, int] = {}
        self.per_client_samples: dict[str, int] = {}
        self.skipped_labels = 0
        self.rounds: list[dict] = []
        self.bytes_broadcast_total = 0

        self.mqtt = MQTTClient(broker=args.broker, port=args.port, node_id="server")

    def _on_data(self, topic: str, payload: bytes) -> None:
        self.inbox.put((topic, payload))

    def connect(self) -> bool:
        if not self.mqtt.connect():
            return False
        return self.mqtt.subscribe(DATA_TOPIC, self._on_data, qos=1)

    def ingest(self, topic: str, payload: bytes) -> None:
        node_id = topic.split("/")[2]
        self.per_client_bytes[node_id] = (
            self.per_client_bytes.get(node_id, 0) + len(payload)
        )

        raw = zlib.decompress(payload)
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        specs = z["spectrograms"]  # (N, 51, 128) float16 — already STFT'd on edge
        metadata = json.loads(str(z["metadata"]))

        labels, keep = [], []
        for j, m in enumerate(metadata):
            idx = self.cls_to_idx.get(m["label"])
            if idx is None:
                self.skipped_labels += 1
                continue
            labels.append(idx)
            keep.append(j)

        if not keep:
            return

        # No STFT here — that's the whole point of Strategy B. Just cast the
        # float16 wire format back up to float32 and add the channel dim.
        specs = specs[keep].astype(np.float32)[..., np.newaxis]
        self.X_pool.append(specs)
        self.y_pool.append(np.asarray(labels, dtype=np.int32))
        self.pool_count += len(labels)
        self.per_client_samples[node_id] = (
            self.per_client_samples.get(node_id, 0) + len(labels)
        )
        print(f"Batch from {node_id}: {len(labels)} spectrograms "
              f"({len(payload) / 1024:.0f} KB compressed) — pool={self.pool_count}",
              flush=True)

    def maybe_train(self) -> None:
        while (len(self.rounds) < self.args.num_rounds
               and self.pool_count >= (len(self.rounds) + 1) * self.args.train_trigger_samples):
            self.run_round()

    def run_round(self) -> None:
        rnd = len(self.rounds) + 1
        X = np.concatenate(self.X_pool)
        y = np.concatenate(self.y_pool)
        print(f"\n=== Round {rnd}/{self.args.num_rounds}: "
              f"training on {len(y)} pooled samples ===", flush=True)

        t0 = time.perf_counter()
        self.model.fit(X, y, epochs=self.args.epochs_per_round,
                       batch_size=self.args.batch_size, verbose=2)
        train_time = time.perf_counter() - t0

        _, test_acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        broadcast_bytes = self.broadcast_model()

        self.rounds.append({
            "round": rnd,
            "buffer_samples": int(len(y)),
            "test_accuracy": float(test_acc),
            "train_time_seconds": round(train_time, 2),
            "broadcast_bytes": broadcast_bytes,
        })
        print(f"=== Round {rnd} done: test_acc={test_acc:.3f}, "
              f"train={train_time:.1f}s, broadcast={broadcast_bytes / 1024:.0f} KB ===\n",
              flush=True)
        self.write_report()

    def broadcast_model(self) -> int:
        buf = io.BytesIO()
        np.savez_compressed(
            buf, **{f"w{i}": w for i, w in enumerate(self.model.get_weights())}
        )
        payload = buf.getvalue()
        self.mqtt.publish(MODEL_TOPIC, payload, qos=1)
        self.bytes_broadcast_total += len(payload)
        return len(payload)

    def write_report(self) -> None:
        report = {
            "strategy": "hybrid",
            "num_rounds_target": self.args.num_rounds,
            "train_trigger_samples": self.args.train_trigger_samples,
            "epochs_per_round": self.args.epochs_per_round,
            "warm_started": self.warm_started,
            "classes": self.classes,
            "rounds": self.rounds,
            "samples_received_total": self.pool_count,
            "skipped_unknown_labels": self.skipped_labels,
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
        self.save_model()
        self.mqtt.disconnect()
        print(f"Server stopped. {len(self.rounds)}/{self.args.num_rounds} rounds, "
              f"{self.pool_count} samples received.", flush=True)
        return 0


def main() -> int:
    args = parse_args()
    server = StrategyBServer(args)

    def handle_term(signum, frame):
        print(f"Received signal {signum} — shutting down", flush=True)
        server.stop_requested = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    return server.run()


if __name__ == "__main__":
    sys.exit(main())
