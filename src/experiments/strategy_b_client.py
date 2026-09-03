"""
Strategy B Client — fake Raspberry Pi (Hybrid: STFT on edge)
============================================================

Identical replay loop to strategy_a_client.py — streams UrbanSound8K
train folds at sensor cadence, disjoint halves for nodes A/B — EXCEPT the
sample is run through HybridStrategy, which does the STFT on-device and
ships a (51,128) float16 spectrogram instead of raw audio.

The bandwidth difference between this and strategy_a_client.py, measured
over the same broker, is Strategy B's whole point.

Spawned by scripts/run_strategy.py --strategy b.

Output: {run-dir}/client_{node_id}.json with upload + STFT-time metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import librosa  # noqa: E402

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.edge.strategies.hybrid import HybridStrategy  # noqa: E402

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 8000  # 500 ms
TRAIN_FOLDS = set(range(1, 9))
MANIFEST_SEED = 42

METADATA = REPO_ROOT / "data/urbansound8k/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_ROOT = REPO_ROOT / "data/urbansound8k/UrbanSound8K/audio"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True, help="Edge node id (A or B)")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--stream-rate-ms", type=int, default=500)
    p.add_argument("--buffer-size", type=int, default=500)
    p.add_argument("--run-dir", required=True)
    return p.parse_args()


def build_manifest(node_id: str) -> list[dict]:
    """Deterministic disjoint split of train folds — same scheme as A, so
    A-vs-B comparison uses identical data per node."""
    with METADATA.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["fold"]) in TRAIN_FOLDS]
    rows.sort(key=lambda r: (int(r["fold"]), r["slice_file_name"]))
    random.Random(MANIFEST_SEED).shuffle(rows)
    if node_id not in ("A", "B"):
        raise SystemExit(f"unsupported --node-id {node_id!r} (expected A or B)")
    offset = 0 if node_id == "A" else 1
    return rows[offset::2]


def load_as_mic_chunk(wav_path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    n = len(audio)
    if n >= CHUNK_SAMPLES:
        start = (n - CHUNK_SAMPLES) // 2
        chunk = audio[start:start + CHUNK_SAMPLES]
    else:
        pad = CHUNK_SAMPLES - n
        chunk = np.pad(audio, (pad // 2, pad - pad // 2))
    return np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.node_id)
    print(f"[client-{args.node_id}] manifest: {len(manifest)} clips "
          f"({'even' if args.node_id == 'A' else 'odd'} half)", flush=True)

    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"[client-{args.node_id}] ERROR: cannot connect to broker "
              f"{args.broker}:{args.port}", file=sys.stderr)
        return 1

    # The one real difference from Strategy A: an STFT processor on-device,
    # feeding HybridStrategy which ships spectrograms.
    stft = STFTProcessor(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)
    strategy = HybridStrategy(
        node_id=args.node_id,
        mqtt_client=mqtt_client,
        stft_processor=stft,
        config={"buffer_size": args.buffer_size},
    )

    rate_s = args.stream_rate_ms / 1000.0
    label_counts: Counter = Counter()
    failed_uploads = 0
    t_start = time.perf_counter()

    for i in range(args.max_samples):
        row = manifest[i % len(manifest)]
        wav = AUDIO_ROOT / f"fold{row['fold']}" / row["slice_file_name"]
        strategy.on_sample({
            "audio": load_as_mic_chunk(wav),
            "label": row["class"],
            "timestamp": time.time(),
            "confidence": 1.0,
        })
        label_counts[row["class"]] += 1

        if strategy.should_trigger():
            if not strategy.trigger():
                failed_uploads += 1
                strategy.clear_buffer()
            m = strategy.get_metrics()
            print(f"[client-{args.node_id}] {i + 1}/{args.max_samples} streamed, "
                  f"batch {m['batches_uploaded']} uploaded, "
                  f"{m['bytes_uploaded'] / 1024 / 1024:.2f} MB total", flush=True)

        next_tick = t_start + (i + 1) * rate_s
        delay = next_tick - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    if strategy._buffer:
        leftover = strategy._buffer.copy()
        strategy._buffer.clear()
        print(f"[client-{args.node_id}] flushing final partial batch "
              f"({len(leftover)} samples)", flush=True)
        if not strategy.execute(leftover):
            failed_uploads += 1

    time.sleep(2.0)
    mqtt_client.disconnect()

    wall = time.perf_counter() - t_start
    metrics = strategy.get_metrics()
    result = {
        "node_id": args.node_id,
        "strategy": "hybrid",
        "samples_streamed": args.max_samples,
        "wall_seconds": round(wall, 2),
        "stream_rate_ms": args.stream_rate_ms,
        "buffer_size": args.buffer_size,
        "batches_uploaded": metrics["batches_uploaded"],
        "bytes_uploaded": metrics["bytes_uploaded"],
        "stft_time_total": round(metrics["stft_time_total"], 2),
        "failed_uploads": failed_uploads,
        "label_counts": dict(label_counts),
    }
    out = run_dir / f"client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[client-{args.node_id}] done: {args.max_samples} samples in {wall:.1f}s, "
          f"{metrics['bytes_uploaded'] / 1024 / 1024:.2f} MB uploaded, "
          f"STFT {metrics['stft_time_total']:.1f}s → {out}", flush=True)
    return 0 if failed_uploads == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
