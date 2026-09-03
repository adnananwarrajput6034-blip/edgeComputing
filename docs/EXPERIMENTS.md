# Experiments — how to reproduce

Exact commands to set up, run each strategy, and read the outputs. Design
context: [ARCHITECTURE.md](ARCHITECTURE.md). Execution internals:
[CODE_FLOW.md](CODE_FLOW.md). Measured numbers: [RESULTS.md](RESULTS.md).

All commands run from the repo root with the venv Python (`.venv/bin/python`).

---

## 0. One-time setup

```bash
# environment (Python 3.13)
python3 -m venv .venv
.venv/bin/pip install -e ".[edge,server,dev]"

# dataset (~6 GB download) + preprocessing cache (data/urbansound8k/cache.npz)
.venv/bin/python scripts/download_urbansound8k.py
.venv/bin/python scripts/prepare_urbansound8k.py

# baseline AudioCNN → models/audio_cnn_urbansound8k.keras  (~90 min)
# Strategies warm-start from this. Skip to run cold (accuracy starts ~random).
.venv/bin/python scripts/train_audio_cnn.py
```

Verify the pipeline pieces (optional, fast):

```bash
.venv/bin/python scripts/test_yolo.py          # labeler
.venv/bin/python scripts/test_stft.py          # audio → (51,128)
.venv/bin/python scripts/test_audio_cnn.py     # model builds + forward pass
.venv/bin/python scripts/test_fusion_model.py  # fusion model builds
```

---

## 1. Start the broker

Science runs (localhost):

```bash
/opt/homebrew/sbin/mosquitto -c configs/mosquitto.conf
```

Real-Pi runs (LAN-reachable, 0.0.0.0):

```bash
/opt/homebrew/sbin/mosquitto -c configs/mosquitto_lan.conf
```

The runner preflights the broker and errors out if it is not reachable.

---

## 2. Run a strategy comparison (science track)

One command spawns 1 server + 2 clients over the broker:

```bash
# Strategy A — centralized (raw audio → server trains)
.venv/bin/python scripts/run_strategy.py --strategy a \
    --max-samples 5000 --stream-rate-ms 500 --num-rounds 10 \
    --buffer-size 500 --train-trigger-samples 1000

# Strategy B — hybrid (edge STFT → spectrograms → server trains)
.venv/bin/python scripts/run_strategy.py --strategy b \
    --max-samples 5000 --num-rounds 10

# Strategy C — federated (local training → FedAvg). Use the tuned local LR.
.venv/bin/python scripts/run_strategy.py --strategy c \
    --max-samples 5000 --num-rounds 10 --epochs-per-round 1 --local-lr 5e-5
```

Smoke test any strategy first (~1 min):

```bash
.venv/bin/python scripts/run_strategy.py --strategy c \
    --max-samples 100 --num-rounds 2 --local-lr 5e-5
```

### Key arguments

| Arg | Meaning | A/B | C |
|---|---|---|---|
| `--strategy` | a / b / c | required | required |
| `--max-samples` | samples streamed per client | stream length | caps local dataset |
| `--stream-rate-ms` | sensor cadence between samples | used | unused (C trains full set) |
| `--buffer-size` | client batch before publish | used | unused |
| `--num-rounds` | server training / FL rounds | used | used |
| `--train-trigger-samples` | pool size that triggers a server round | used | unused |
| `--epochs-per-round` | — | server fit epochs | **local** epochs per client |
| `--local-lr` | client local learning rate | n/a | **use 5e-5** (1e-3 destabilizes) |

### Fusion-era strategies (F6 — needs the VGGSound paired cache)

```bash
# one-time: precompute vision FVs + build/verify the FV-fusion model
.venv/bin/python scripts/prepare_fusion_fv.py

# FB — hybrid with fusion (edge ships spectrogram + FV)
.venv/bin/python scripts/run_strategy.py --strategy fb \
    --buffer-size 250 --train-trigger-samples 500 --num-rounds 6 --stream-rate-ms 50

# FC — federated fusion (weights only; gentle local LR)
.venv/bin/python scripts/run_strategy.py --strategy fc --num-rounds 10 --local-lr 1e-4
```

Servers log clean AND blackout accuracy per round. Prereqs: paired cache +
trained fusion model (see FUSION.md Parts 2–3).

### Reproduce the Strategy C local-LR sweep

```bash
for lr in 1e-3 5e-4 1e-4 5e-5; do
  .venv/bin/python scripts/run_strategy.py --strategy c \
      --max-samples 5000 --num-rounds 10 --local-lr $lr
done
```

Expect accuracy to stabilize as LR drops (see RESULTS.md §4).

---

## 3. Run on a real Pi (hardware track)

On the **laptop**: start the LAN broker and the server (unchanged from science):

```bash
/opt/homebrew/sbin/mosquitto -c configs/mosquitto_lan.conf
.venv/bin/python -m src.experiments.strategy_a_server \
    --broker localhost --num-rounds 1 --train-trigger-samples 500 \
    --run-dir results/pi_capture_server
```

On the **Pi** (repo cloned, `.venv` with edge deps), stream live capture to the
laptop's IP:

```bash
.venv/bin/python -m src.experiments.strategy_a_pi_client \
    --node-id A --broker <LAPTOP_LAN_IP> \
    --buffer-size 25 --audio-device <mic_index> --duration-min 2 \
    --save-evidence
```

- Find the mic index: `python -m sounddevice` (pick the USB audio input).
- `--save-evidence` writes frame+audio+label triplets + an `index.html` gallery
  under `results/pi_capture/evidence_*` (demo only; production never persists frames).

---

## 4. Outputs

```
results/strategy_{a,b,c}/run_<timestamp>/
    server.json      rounds[] (round, buffer_samples, test_accuracy,
                     train_time_seconds), per_client_bytes_total,
                     bytes_broadcast_total, warm_started
    client_A.json    bytes_uploaded, batches/rounds, timings, label_counts
    client_B.json    (B adds stft_time_total; C adds round_times)
    server.log, client_*.log
    invocation.txt   exact command + args (reproducibility)

results/pi_capture/
    pi_client_A.json ticks, detection_rate, yolo_ms_mean, bytes_uploaded
    evidence_<ts>/   frame+wav+label triplets + manifest.json + index.html
```

### Reading a result

```bash
# round-by-round accuracy + bandwidth for a run
.venv/bin/python - <<'PY'
import json, glob
run = sorted(glob.glob('results/strategy_c/run_*'))[-1]
d = json.load(open(run + '/server.json'))
for r in d['rounds']:
    print(f"round {r['round']:2d}  acc={r['test_accuracy']:.3f}")
up = d['per_client_bytes_total']
print('upload/node MB:', {k: round(v/1024/1024, 2) for k, v in up.items()})
PY
```

Sender vs receiver byte counts (`client_*.json` `bytes_uploaded` vs
`server.json` `per_client_bytes_total`) should match — the cross-check that
makes the bandwidth numbers trustworthy.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `cannot reach MQTT broker` | start mosquitto (§1); for Pi use `mosquitto_lan.conf` + laptop LAN IP |
| Server never prints "ready" | check `server.log`; usually missing `cache.npz` (run `prepare_urbansound8k.py`) |
| C accuracy oscillates / drops | local LR too high — use `--local-lr 5e-5` |
| `skipped_unknown_labels` high (Pi) | live YOLO labels (person, car) aren't UrbanSound8K classes — expected for the demo |
| Pi mic silent / full-scale constant | dead webcam mic — use a separate USB mic; verify with `python -m sounddevice` + an RMS check |
