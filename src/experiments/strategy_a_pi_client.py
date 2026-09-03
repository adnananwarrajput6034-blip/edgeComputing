"""
Strategy A Pi Client — REAL capture (webcam + webcam mic + YOLO)
================================================================

The real-hardware counterpart of strategy_a_client.py. Same downstream
pipeline (CentralizedStrategy → MQTT → laptop server); only the sample
source differs — live sensors instead of UrbanSound8K replay:

    every tick (~1s):
        start 500ms mic recording (webcam mic, native rate → 16 kHz int16)
        grab a frame mid-recording (image sits inside the audio window)
        YOLO labels the frame (~430ms on Pi 5)
        detection → {audio, label, confidence, timestamp} → buffer
        nothing detected → tick skipped (recall behavior: skipped, not
        mislabeled — same as the VGGSound validation)

    buffer full → CentralizedStrategy zips batch → MQTT → laptop server.
    Frames are DISCARDED after YOLO — images never leave the Pi.

Run on the Pi:

    .venv/bin/python -m src.experiments.strategy_a_pi_client \
        --node-id A --broker <LAPTOP_IP> --buffer-size 25 --audio-device 1

Live labels are COCO classes (person, dog, car, ...). The laptop server
maps labels against UrbanSound8K's 10 classes, so most live samples land
in skipped_unknown_labels — expected for the demo; the flow + bandwidth
numbers are the point.

Stop with Ctrl-C (or --duration-min): flushes the partial batch and writes
{run-dir}/pi_client_{node_id}.json.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import librosa  # noqa: E402
import sounddevice as sd  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.edge.processing.vision import VisionProcessor  # noqa: E402
from src.edge.strategies.centralized import CentralizedStrategy  # noqa: E402

TARGET_SR = 16000
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)  # 8000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True, help="Edge node id (A or B)")
    p.add_argument("--broker", required=True, help="Laptop IP running mosquitto")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--buffer-size", type=int, default=25,
                   help="Samples per upload batch (small → frequent demo uploads)")
    p.add_argument("--tick-ms", type=int, default=1000,
                   help="Capture cadence; 500ms audio + ~430ms YOLO fit in 1s")
    p.add_argument("--duration-min", type=float, default=None,
                   help="Stop after N minutes (default: run until Ctrl-C)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--audio-device", type=int, default=None,
                   help="sounddevice input index (list: python -m sounddevice)")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights (auto-downloaded if absent)")
    p.add_argument("--run-dir", default="results/pi_capture")
    p.add_argument("--dataset", default=None,
                   help="Replay an enrolled dataset from pi_enroll.py instead "
                        "of capturing live. Strategies A/B/C must replay the "
                        "SAME dataset for their bandwidth numbers to be "
                        "comparable — different samples, different bytes.")
    p.add_argument("--shard", default=None,
                   help="'i/n' — send only this node's slice (replay mode)")
    p.add_argument("--save-evidence", action="store_true",
                   help="DEMO/DEBUG ONLY: additionally save frame.jpg + "
                        "audio.wav + label for every labeled tick, so humans "
                        "can inspect what YOLO saw and the mic heard. The "
                        "production pipeline never persists or ships frames.")
    return p.parse_args()


class MicRecorder:
    """500ms chunks from the webcam mic at its native rate, resampled to
    16 kHz int16 — USB mics rarely support 16 kHz directly."""

    def __init__(self, device: int | None):
        self.device = device
        info = sd.query_devices(device, "input")
        self.native_sr = int(info["default_samplerate"])
        self.name = info["name"]

    def start_chunk(self):
        frames = int(self.native_sr * CHUNK_SECONDS)
        return sd.rec(frames, samplerate=self.native_sr, channels=1,
                      dtype="float32", device=self.device)

    def finish(self, recording) -> np.ndarray:
        sd.wait()
        audio = recording[:, 0]
        if self.native_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=self.native_sr,
                                     target_sr=TARGET_SR)
        if len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        audio = audio[:CHUNK_SAMPLES]
        return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


def run_replay(args, run_dir: Path) -> int:
    """Replay an enrolled dataset as raw audio — Strategy A's defining trait.

    Same samples as the B and C live clients, so the only thing that differs
    across the three reports is what each strategy puts on the wire.
    """
    from src.experiments.pi_live import load_dataset

    ds = load_dataset(Path(args.dataset))
    audio, y, classes = ds["audio_train"], ds["y_train"], ds["classes"]

    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        audio, y = audio[i::n], y[i::n]
        print(f"[pi-{args.node_id}] shard {args.shard}: {len(audio)} samples",
              flush=True)

    print(f"[pi-{args.node_id}] dataset: {len(audio)} samples, "
          f"classes={classes}", flush=True)

    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"ERROR: cannot reach broker {args.broker}:{args.port}",
              file=sys.stderr)
        return 1

    strategy = CentralizedStrategy(
        node_id=args.node_id,
        mqtt_client=mqtt_client,
        config={"buffer_size": args.buffer_size},
    )

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    label_counts: Counter = Counter()
    t_start = time.perf_counter()
    print(f"[pi-{args.node_id}] streaming raw audio (Ctrl-C to stop)...",
          flush=True)

    for i in range(len(audio)):
        if stop["flag"]:
            break
        label = classes[int(y[i])]
        label_counts[label] += 1
        strategy.on_sample({
            "audio": audio[i],
            "label": label,
            "timestamp": time.time(),
            "confidence": 1.0,
        })
        if strategy.should_trigger():
            strategy.trigger()
            m = strategy.get_metrics()
            print(f"[pi-{args.node_id}] batch {m['batches_uploaded']} uploaded "
                  f"({m['bytes_uploaded'] / 1024:.0f} KB total)", flush=True)

    if strategy._buffer:
        leftover = strategy._buffer.copy()
        strategy._buffer.clear()
        print(f"[pi-{args.node_id}] flushing final partial batch "
              f"({len(leftover)} samples)", flush=True)
        strategy.execute(leftover)

    time.sleep(2.0)  # drain QoS 1 deliveries
    mqtt_client.disconnect()

    metrics = strategy.get_metrics()
    result = {
        "node_id": args.node_id,
        "strategy": "A_centralized",
        "mode": "live_vocabulary_replay",
        "dataset": str(args.dataset),
        "classes": classes,
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "samples_sent": int(len(audio)),
        "batches_uploaded": metrics["batches_uploaded"],
        "bytes_uploaded": metrics["bytes_uploaded"],
        "label_counts": dict(label_counts),
    }
    out = run_dir / f"pi_client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[pi-{args.node_id}] done: {len(audio)} samples, "
          f"{metrics['bytes_uploaded'] / 1024:.0f} KB uploaded -> {out}",
          flush=True)
    return 0


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        return run_replay(args, run_dir)

    print(f"[pi-{args.node_id}] opening camera {args.camera_index}...", flush=True)
    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        print("ERROR: cannot open camera", file=sys.stderr)
        return 1

    mic = MicRecorder(args.audio_device)
    print(f"[pi-{args.node_id}] mic: {mic.name} @ {mic.native_sr} Hz "
          f"→ {TARGET_SR} Hz", flush=True)

    print(f"[pi-{args.node_id}] loading YOLO ({args.model})...", flush=True)
    vision = VisionProcessor(model_path=args.model,
                             confidence_threshold=args.confidence)
    # Warmup: first inference pays ~4s of graph init — do it on a throwaway
    # frame now so tick 1 runs at steady-state speed.
    ok, warm = cam.read()
    if ok:
        vision.detect(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB))
    print(f"[pi-{args.node_id}] YOLO warmed up", flush=True)

    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"ERROR: cannot reach broker {args.broker}:{args.port}",
              file=sys.stderr)
        return 1

    strategy = CentralizedStrategy(
        node_id=args.node_id,
        mqtt_client=mqtt_client,
        config={"buffer_size": args.buffer_size},
    )

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    evidence_dir = None
    evidence_manifest: list[dict] = []
    if args.save_evidence:
        evidence_dir = run_dir / f"evidence_{time.strftime('%Y%m%d_%H%M%S')}"
        evidence_dir.mkdir(parents=True)
        print(f"[pi-{args.node_id}] EVIDENCE MODE: saving labeled ticks "
              f"to {evidence_dir}", flush=True)

    ticks = 0
    labeled = 0
    label_counts: Counter = Counter()
    yolo_ms: list[float] = []
    t_start = time.perf_counter()
    deadline = (t_start + args.duration_min * 60) if args.duration_min else None

    print(f"[pi-{args.node_id}] capturing (Ctrl-C to stop)...", flush=True)
    while not stop["flag"]:
        if deadline and time.perf_counter() > deadline:
            break
        tick_t0 = time.perf_counter()
        ticks += 1

        rec = mic.start_chunk()          # 500ms recording begins
        ok, frame_bgr = cam.read()       # frame lands mid-window
        audio = mic.finish(rec)

        if not ok:
            print("WARN: camera read failed, skipping tick", flush=True)
            continue

        y0 = time.perf_counter()
        label, confidence, _ = vision.detect_and_label(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        yolo_ms.append((time.perf_counter() - y0) * 1000)

        if label is not None:
            labeled += 1
            label_counts[label] += 1
            strategy.on_sample({
                "audio": audio,
                "label": label,
                "timestamp": time.time(),
                "confidence": confidence,
            })
            print(f"[pi-{args.node_id}] tick {ticks}: {label} ({confidence:.2f}) "
                  f"— buffer {strategy.get_buffer_status()['current_size']}"
                  f"/{args.buffer_size}", flush=True)
            if evidence_dir is not None:
                import soundfile as sf
                stem = f"tick{ticks:04d}_{label.replace(' ', '_')}"
                cv2.imwrite(str(evidence_dir / f"{stem}.jpg"), frame_bgr)
                sf.write(str(evidence_dir / f"{stem}.wav"), audio, TARGET_SR)
                evidence_manifest.append({
                    "tick": ticks, "label": label,
                    "confidence": round(confidence, 3),
                    "frame": f"{stem}.jpg", "audio": f"{stem}.wav",
                })
            if strategy.should_trigger():
                strategy.trigger()
                m = strategy.get_metrics()
                print(f"[pi-{args.node_id}] batch {m['batches_uploaded']} uploaded "
                      f"({m['bytes_uploaded'] / 1024:.0f} KB total)", flush=True)

        delay = args.tick_ms / 1000.0 - (time.perf_counter() - tick_t0)
        if delay > 0:
            time.sleep(delay)

    # flush partial final batch (same private-buffer note as replay client)
    if strategy._buffer:
        leftover = strategy._buffer.copy()
        strategy._buffer.clear()
        print(f"[pi-{args.node_id}] flushing final partial batch "
              f"({len(leftover)} samples)", flush=True)
        strategy.execute(leftover)

    time.sleep(2.0)  # drain QoS 1 deliveries
    mqtt_client.disconnect()
    cam.release()

    wall = time.perf_counter() - t_start
    metrics = strategy.get_metrics()
    result = {
        "node_id": args.node_id,
        "mode": "live_capture",
        "wall_seconds": round(wall, 1),
        "ticks": ticks,
        "labeled_ticks": labeled,
        "detection_rate": round(labeled / max(ticks, 1), 3),
        "yolo_ms_mean": round(float(np.mean(yolo_ms)), 1) if yolo_ms else None,
        "batches_uploaded": metrics["batches_uploaded"],
        "bytes_uploaded": metrics["bytes_uploaded"],
        "label_counts": dict(label_counts),
    }
    out = run_dir / f"pi_client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    if evidence_dir is not None:
        (evidence_dir / "manifest.json").write_text(
            json.dumps(evidence_manifest, indent=2))
        print(f"[pi-{args.node_id}] evidence: {len(evidence_manifest)} triplets "
              f"in {evidence_dir}", flush=True)
    print(f"[pi-{args.node_id}] done: {ticks} ticks, {labeled} labeled "
          f"({result['detection_rate']:.0%}), "
          f"{metrics['bytes_uploaded'] / 1024:.0f} KB uploaded → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
