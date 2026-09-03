"""
Strategy B Pi Client — STFT on the edge, spectrograms uploaded
==============================================================

The real-hardware counterpart of strategy_b_client.py, and the live-track
sibling of strategy_a_pi_client.py. Same enrolled samples, same broker,
same server-side training — the ONLY difference from Strategy A is where
the STFT runs:

    A:  Pi sends raw int16 audio        -> server does STFT, then trains
    B:  Pi does STFT, sends spectrogram -> server trains directly

That single move is what the strategy comparison measures. Doing it on the
Pi costs edge CPU and buys bandwidth (float16 spectrograms compress far
better than raw waveforms) and privacy (a mel-spectrogram is much harder to
reconstruct speech from than the waveform it came from).

Replays the vocabulary enrolled by pi_enroll.py, so A, B and C all train on
exactly the same samples and the bandwidth numbers are comparable.

Run on the Pi:

    .venv/bin/python -m src.experiments.strategy_b_pi_client \
        --node-id A --broker <LAPTOP_LAN_IP> \
        --dataset results/live/live_dataset.npz --buffer-size 25

Writes {--run-dir}/pi_client_{node_id}.json with bytes_uploaded and the
edge STFT time, which is the cost side of the trade.
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

from src.edge.communication.mqtt_client import MQTTClient  # noqa: E402
from src.edge.processing.stft import STFTProcessor  # noqa: E402
from src.edge.strategies.hybrid import HybridStrategy  # noqa: E402
from src.experiments.pi_live import load_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--node-id", required=True, help="Edge node id (A or B)")
    p.add_argument("--broker", required=True, help="Laptop IP running mosquitto")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--dataset", default="results/live/live_dataset.npz",
                   help="Enrolled dataset from pi_enroll.py")
    p.add_argument("--buffer-size", type=int, default=25,
                   help="Samples per upload batch")
    p.add_argument("--tick-ms", type=int, default=0,
                   help="Pace replay to imitate live capture (0 = as fast as "
                        "possible, which is what you want in a demo)")
    p.add_argument("--shard", default=None,
                   help="'i/n' — send only this node's slice of the data, so "
                        "two Pis can act as two non-overlapping clients")
    p.add_argument("--run-dir", default="results/live/strategy_b")
    return p.parse_args()


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

    print(f"[pi-{args.node_id}] dataset: {len(audio)} samples, "
          f"classes={classes}", flush=True)

    # Same STFT parameters as the server and prepare_urbansound8k.py, so
    # edge-computed spectrograms land in the server's feature space.
    stft = STFTProcessor(sample_rate=16000, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)

    mqtt_client = MQTTClient(broker=args.broker, port=args.port,
                             node_id=args.node_id)
    if not mqtt_client.connect():
        print(f"ERROR: cannot reach broker {args.broker}:{args.port}",
              file=sys.stderr)
        return 1

    strategy = HybridStrategy(
        node_id=args.node_id,
        mqtt_client=mqtt_client,
        stft_processor=stft,
        config={"buffer_size": args.buffer_size},
    )

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    label_counts: Counter = Counter()
    t_start = time.perf_counter()

    print(f"[pi-{args.node_id}] streaming spectrograms (Ctrl-C to stop)...",
          flush=True)
    for i in range(len(audio)):
        if stop["flag"]:
            break
        tick_t0 = time.perf_counter()
        label = classes[int(y[i])]
        label_counts[label] += 1

        # STFT happens inside HybridStrategy.prepare_sample — on the Pi.
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

        if args.tick_ms:
            delay = args.tick_ms / 1000.0 - (time.perf_counter() - tick_t0)
            if delay > 0:
                time.sleep(delay)

    if strategy._buffer:
        leftover = strategy._buffer.copy()
        strategy._buffer.clear()
        print(f"[pi-{args.node_id}] flushing final partial batch "
              f"({len(leftover)} samples)", flush=True)
        strategy.execute(leftover)

    time.sleep(2.0)  # drain QoS 1 deliveries before dropping the socket
    mqtt_client.disconnect()

    metrics = strategy.get_metrics()
    result = {
        "node_id": args.node_id,
        "strategy": "B_hybrid",
        "mode": "live_vocabulary_replay",
        "dataset": str(args.dataset),
        "classes": classes,
        "wall_seconds": round(time.perf_counter() - t_start, 1),
        "samples_sent": int(len(audio)),
        "batches_uploaded": metrics["batches_uploaded"],
        "bytes_uploaded": metrics["bytes_uploaded"],
        "stft_time_total": round(metrics.get("stft_time_total", 0.0), 2),
        "label_counts": dict(label_counts),
    }
    out = run_dir / f"pi_client_{args.node_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[pi-{args.node_id}] done: {len(audio)} samples, "
          f"{metrics['bytes_uploaded'] / 1024:.0f} KB uploaded, "
          f"edge STFT {result['stft_time_total']}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
