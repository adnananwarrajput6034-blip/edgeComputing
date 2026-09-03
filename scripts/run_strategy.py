"""
Strategy Harness Runner (generalized over A/B)
==============================================

Spawns 1 server + 2 client subprocesses that communicate over a real MQTT
broker (mosquitto on localhost:1883) and stream UrbanSound8K samples for a
chosen strategy. Generalizes run_strategy_a.py with a --strategy flag:

    --strategy a  → src.experiments.strategy_a_{server,client}  (centralized,
                    raw audio, server-side STFT)
    --strategy b  → src.experiments.strategy_b_{server,client}  (hybrid,
                    edge-side STFT, spectrograms on the wire)

Both write the same server.json / client_*.json schema, into
results/strategy_{a,b,c}/run_{timestamp}/ — so results sit side by side
for the A-vs-B-vs-C comparison.

Prerequisites: mosquitto running (configs/mosquitto.conf), venv, and
data/urbansound8k/cache.npz.

Usage:
    # smoke test (~1 min)
    .venv/bin/python scripts/run_strategy.py --strategy b \\
        --max-samples 100 --stream-rate-ms 100 --num-rounds 2 \\
        --buffer-size 50 --train-trigger-samples 100

    # real run (~40 min)
    .venv/bin/python scripts/run_strategy.py --strategy b \\
        --max-samples 5000 --num-rounds 10
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", choices=["a", "b", "c", "fb", "fc"],
                   required=True,
                   help="a=centralized (raw audio), b=hybrid (spectrograms), "
                        "c=federated (FedAvg) — audio model; "
                        "fb/fc = same with the FUSION model (F6: paired "
                        "VGGSound data, edge ships/keeps the vision FV)")
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Samples per client")
    p.add_argument("--stream-rate-ms", type=int, default=500,
                   help="Sleep between samples (500ms = real sensor cadence)")
    p.add_argument("--buffer-size", type=int, default=500,
                   help="Client-side batch size before MQTT publish")
    p.add_argument("--num-rounds", type=int, default=10,
                   help="Number of server training rounds to run")
    p.add_argument("--train-trigger-samples", type=int, default=1000,
                   help="Server-side buffer size that triggers a training round")
    p.add_argument("--epochs-per-round", type=int, default=1)
    p.add_argument("--local-lr", type=float, default=1e-3,
                   help="Strategy C only: local learning rate for FedAvg clients")
    p.add_argument("--server-ready-timeout", type=float, default=60.0,
                   help="Max seconds to wait for server to subscribe before starting clients")
    p.add_argument("--client-timeout", type=float, default=None,
                   help="Kill clients after N seconds (safety net; default = no timeout)")
    return p.parse_args()


def check_broker(host: str, port: int, timeout: float = 3.0) -> bool:
    """Try a TCP connect to the broker to make sure mosquitto is up."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def spawn(name: str, cmd: list, log_path: Path) -> subprocess.Popen:
    """Spawn a subprocess with stdout+stderr redirected to a log file."""
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
    )
    proc._log_fh = log_fh  # keep ref so it doesn't get GC'd
    print(f"  → spawned {name} (pid={proc.pid}), logging to {log_path}")
    return proc


def main() -> int:
    args = parse_args()
    server_module = f"src.experiments.strategy_{args.strategy}_server"
    client_module = f"src.experiments.strategy_{args.strategy}_client"

    # Preflight
    if not VENV_PYTHON.exists():
        print(f"ERROR: venv python not found at {VENV_PYTHON}", file=sys.stderr)
        return 1

    if not check_broker(args.broker, args.port):
        print(
            f"ERROR: cannot reach MQTT broker at {args.broker}:{args.port}.\n"
            f"Start mosquitto first:\n"
            f"    brew services start mosquitto\n"
            f"  OR\n"
            f"    mosquitto -v  (in another terminal, to see traffic)",
            file=sys.stderr,
        )
        return 1

    # Run directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "results" / f"strategy_{args.strategy}" / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Strategy {args.strategy.upper()} — run directory: {run_dir}")

    # Write the invocation for reproducibility
    with open(run_dir / "invocation.txt", "w") as f:
        f.write("argv: " + " ".join(sys.argv) + "\n")
        f.write(f"timestamp: {ts}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # 1) Server first (so it's subscribed before clients start publishing)
    print("\nStarting server...")
    server_cmd = [
        str(VENV_PYTHON), "-m", server_module,
        "--broker", args.broker,
        "--port", str(args.port),
        "--num-rounds", str(args.num_rounds),
        "--train-trigger-samples", str(args.train_trigger_samples),
        "--epochs-per-round", str(args.epochs_per_round),
        "--run-dir", str(run_dir),
    ]
    if args.strategy in ("c", "fc"):
        server_cmd += ["--local-lr", str(args.local_lr)]  # A/B servers don't accept it
    server_proc = spawn("server", server_cmd, run_dir / "server.log")

    # Wait for server to actually subscribe to the data topic before starting
    # clients — otherwise QoS 1 messages sent before subscription are silently
    # dropped (broker doesn't retain unless retain=True is set).
    ready_marker = "Server ready. Waiting for edge batches"
    server_log = run_dir / "server.log"
    print(f"  waiting up to {args.server_ready_timeout}s for server to become ready...")
    deadline = time.time() + args.server_ready_timeout
    ready = False
    while time.time() < deadline:
        if server_proc.poll() is not None:
            print(f"ERROR: server exited (code {server_proc.returncode})."
                  f" Check {server_log}", file=sys.stderr)
            return 1
        if server_log.exists():
            with open(server_log, "rb") as f:
                if ready_marker.encode() in f.read():
                    ready = True
                    break
        time.sleep(0.5)

    if not ready:
        print(f"ERROR: server did not become ready within "
              f"{args.server_ready_timeout}s. Check {server_log}", file=sys.stderr)
        server_proc.terminate()
        return 1

    print("  server ready ✓")

    # 2) Clients
    print("\nStarting clients...")
    client_procs = []
    for node_id in ["A", "B"]:
        cmd = [
            str(VENV_PYTHON), "-m", client_module,
            "--node-id", node_id,
            "--broker", args.broker,
            "--port", str(args.port),
            "--max-samples", str(args.max_samples),
            "--stream-rate-ms", str(args.stream_rate_ms),
            "--buffer-size", str(args.buffer_size),
            "--run-dir", str(run_dir),
        ]
        proc = spawn(f"client-{node_id}", cmd, run_dir / f"client_{node_id}.log")
        client_procs.append((node_id, proc))
        time.sleep(0.5)  # stagger client startup slightly

    # 3) Wait for clients to finish
    print("\nWaiting for clients to finish streaming...")
    start = time.time()
    for node_id, proc in client_procs:
        rc = proc.wait(timeout=args.client_timeout)
        elapsed = time.time() - start
        print(f"  client-{node_id} exited (code={rc}) after {elapsed:.1f}s")

    # 4) Give server a moment to process the final batch, then signal it to stop
    print("\nDraining server (10s)...")
    time.sleep(10.0)
    if server_proc.poll() is None:
        print("  sending SIGTERM to server")
        server_proc.terminate()
        # Grace period must cover a full training round: if the last client
        # batch tips the pool over the final trigger, the server is inside
        # model.fit() when SIGTERM lands and needs ~30-60s to finish it.
        try:
            server_proc.wait(timeout=120.0)
        except subprocess.TimeoutExpired:
            print("  server didn't exit in 120s, killing")
            server_proc.kill()
            server_proc.wait()

    print(f"\nDone. Artifacts in {run_dir}")
    # Print quick summary if server.json exists
    server_json = run_dir / "server.json"
    if server_json.exists():
        import json
        with open(server_json) as f:
            data = json.load(f)
        rounds = data.get("rounds", [])
        print(f"\nServer completed {len(rounds)}/{data.get('num_rounds_target')} rounds")
        for r in rounds:
            print(
                f"  round {r['round']:2d}: buffer={r['buffer_samples']:5d}, "
                f"test_acc={r['test_accuracy']:.3f}, "
                f"train_time={r['train_time_seconds']:.1f}s"
            )
        print(f"\nBytes broadcast: {data.get('bytes_broadcast_total', 0)/1024:.1f} KB")
        for node, bytes_up in data.get("per_client_bytes_total", {}).items():
            print(f"Bytes received from {node}: {bytes_up/1024/1024:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
