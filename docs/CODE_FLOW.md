# Code Flow — how the system runs

How the code executes end to end: what you launch, what each file does, and
how the flow differs between **simulation** (laptop) and **real Pi** hardware.

There are three levels:
1. **Orchestrator** — the one script you launch.
2. **Strategy modules** — the per-strategy logic (who does what, where).
3. **Shared spine** — MQTT + serialization + models, identical everywhere.

Mental model: *orchestrator launches processes; strategy modules define
who-does-what-where; the spine is constant.* A/B/C differ only in **where STFT
and training run and what crosses the wire**; simulation vs real Pi differ only
in **where samples and labels come from**.

---

## 1. What you actually run

One command launches three processes that talk over the mosquitto broker
(localhost:1883 in simulation):

```
scripts/run_strategy.py --strategy c        ← YOU run this (orchestrator)
        │  spawns 3 subprocesses
        ├── python -m src.experiments.strategy_c_server          (1 server)
        ├── python -m src.experiments.strategy_c_client --node-id A  (fake Pi 1)
        └── python -m src.experiments.strategy_c_client --node-id B  (fake Pi 2)
```

`scripts/run_strategy.py` contains **no learning logic** — it is only a
launcher. Its `main()`:

1. Preflight: venv exists, broker reachable (`check_broker`).
2. Create `results/strategy_<x>/run_<timestamp>/`.
3. Start the **server**, then **wait** until it logs
   `"Server ready. Waiting for edge batches"` before starting clients.
   *Why:* MQTT QoS-1 messages published before the server has subscribed are
   silently dropped — clients must not start early.
4. Start the two **clients**.
5. Wait for clients to finish → drain server 10 s → SIGTERM it → print the
   summary from `server.json`.

`--strategy a|b|c` just selects which module names to spawn
(`strategy_{a,b,c}_{server,client}`). For Strategy C it also passes
`--local-lr` to the server.

---

## 2. Simulation flow — Strategy A (Centralized, the simplest)

**Client** — `src/experiments/strategy_a_client.py`, pretends to be a Pi:

```
build_manifest(node)        → this node's half of UrbanSound8K train folds
                              (deterministic shuffle; A=even, B=odd → disjoint)
loop max_samples times, one every stream_rate_ms:
    load_as_mic_chunk(wav)  → 16 kHz int16, 8000 samples ("what the mic heard")
    sample = {audio, label (from CSV), timestamp}
    CentralizedStrategy.on_sample(sample)      → buffers RAW audio
    when buffer hits buffer_size:
        strategy.trigger() → execute() → serialize (int16 + JSON meta, zlib)
                                        → MQTT publish → thesis/edge/A/data
write client_A.json  (bytes uploaded, batches, label counts)
```

Buffering/serialize/publish live in `src/edge/strategies/centralized.py`
(`CentralizedStrategy`), not the client — the client only feeds it samples.

**Server** — `src/experiments/strategy_a_server.py`:

```
warm-start AudioCNN from models/audio_cnn_urbansound8k.keras
subscribe thesis/edge/+/data          ("+" matches both A and B)
   callback: put(payload) on a queue, return fast   ← producer-consumer:
                                                       training must NOT block
                                                       the network thread
print "Server ready. Waiting for edge batches"       ← orchestrator gate
main loop:
    dequeue batch → ingest(): zlib decompress → np.load → (N,8000) int16
                              → STFT each → spectrograms → add to pool
                              (count bytes per client)
    maybe_train(): when pool ≥ threshold → run_round():
                     model.fit(pool, 1 epoch) → evaluate on fold-10 test set
                     → broadcast weights to thesis/server/model/global
                     → append round to server.json
```

Message flow:

```
client A ─ raw audio batch ─►┌────────┐─► server: STFT → pool → fit → eval
client B ─ raw audio batch ─►│ broker │
                             └────────┘◄─ broadcast weights ─┘
```

---

## 3. Simulation flow — Strategy B (Hybrid): one change

Identical to A, except STFT moves from server to client:

- `strategy_b_client.py` uses `HybridStrategy` (`src/edge/strategies/hybrid.py`),
  whose `prepare_sample` runs STFT → ships a `(51,128)` **float16 spectrogram**
  instead of raw audio.
- `strategy_b_server.py` `ingest()` **skips STFT** — spectrograms arrive ready
  to pool.

That is the entire A→B difference: STFT relocates to the edge; spectrograms
replace raw audio on the wire.

---

## 4. Simulation flow — Strategy C (Federated): round-synchronized

The server stops training; the clients start. Lock-step rounds, not a
continuous stream.

**Server** — `src/experiments/strategy_c_server.py`:

```
warm-start global model; create FedAvgAggregator
subscribe thesis/edge/+/weights
broadcast_global(round 1, retain=True) → thesis/server/model/global
print "Server ready. Waiting for edge batches"
loop until num_rounds:
    collect weights from BOTH clients (round-tagged; stale rounds ignored)
    aggregate_round(): FedAvg weighted-average → set_weights → eval on test
                       → record round → broadcast next global (or done=True)
```

The server **never calls model.fit** — only `FedAvgAggregator.aggregate()`
(`src/server/aggregation/fedavg.py`). ~0.5 s/round.

**Client** — `src/experiments/strategy_c_client.py`:

```
build local dataset ONCE (its half → STFT → X_local, y_local)
build AudioCNN; subscribe thesis/server/model/global
loop:
    receive global (round, weights, local_epochs, local_lr, done)
    if done: break
    model.set_weights(global) → fresh Adam(local_lr) → model.fit(X_local)
    publish weights + n_samples → thesis/edge/A/weights
write client_A.json (local train times, bytes)
```

Message flow (weights both directions; data never moves):

```
server ─ global model (retained) ─►┌────────┐─► client: set_weights → train
                                   │ broker │
server: FedAvg(both) → next global │        │◄─ trained weights ─┘
                                   └────────┘   ... repeat N rounds ...
```

Two design details:
- **Retained** global broadcast → a client that subscribes late still gets the
  current model (no startup race).
- Every message carries a **round number** → the server ignores stragglers
  from a past round.
- **Fresh optimizer each round:** standard FedAvg does not carry optimizer
  state across rounds. Reusing Adam's momentum after `set_weights()` corrupts
  the first steps (this caused an early accuracy collapse until fixed).

---

## 5. Real-Pi flow — what changes (and what does not)

Only the **client's sensor front-end** changes. Everything downstream is
identical.

```
                   SIMULATION                    REAL PI
sample source:  load_as_mic_chunk(wav)   →   MicRecorder (USB mic, 500 ms)
                                              + cv2 frame grab
label source:   row["class"] (CSV)       →   VisionProcessor.detect (YOLO on frame)
──────────────────────────────────────────────────────────────────────
buffer/serialize/publish  ───────── IDENTICAL (CentralizedStrategy) ─────────
MQTT transport            ───────── IDENTICAL (broker = laptop LAN IP) ───────
server                    ───────── IDENTICAL (strategy_a_server.py) ─────────
```

On the Pi you run the client directly:

```bash
python -m src.experiments.strategy_a_pi_client --node-id A --broker <laptop-IP>
```

Its loop (`src/experiments/strategy_a_pi_client.py`): record 500 ms + grab a
frame → YOLO labels the frame → if something is detected, feed
`{audio, label, confidence}` into the **same** `CentralizedStrategy` → same
batch → same broker → the **same unchanged server** on the laptop. The server
cannot tell a real Pi from a fake one — that is the payoff of the shared spine.

- Broker binding switches from `localhost` (`configs/mosquitto.conf`) to
  `0.0.0.0` (`configs/mosquitto_lan.conf`) so the Pi can reach it over the LAN.
- Frames are **discarded after labeling** — images never ship.
- `--save-evidence` additionally saves frame+audio+label triplets for a human
  demo gallery (debug only; the production path never persists frames).

Only Strategy A has a Pi client so far. B/C Pi clients would follow the same
pattern: swap the sensor source, reuse everything else.

---

## 6. The shared spine (identical across strategies and sim/real)

| Component | File | Role |
|---|---|---|
| MQTT wrapper | `src/edge/communication/mqtt_client.py` | connect / publish / subscribe (wildcard dispatch) |
| Broker | mosquitto (`configs/mosquitto*.conf`) | the "post office"; opaque to payloads |
| Serialization | (inline in strategies) | `np.savez_compressed` + `zlib` |
| STFT | `src/edge/processing/stft.py` | audio → `(51,128)` mel-spectrogram |
| Model | `src/server/training/models/audio_cnn.py` | AudioCNN (111K params) |
| Aggregation | `src/server/aggregation/fedavg.py` | weighted weight-average (Strategy C) |
| Labeler (Pi) | `src/edge/processing/vision.py` | YOLOv8 wrapper (`VisionProcessor`) |

---

## 7. MQTT topics

| Topic | Direction | Used by |
|---|---|---|
| `thesis/edge/{id}/data` | edge → server | A, B (raw audio / spectrograms) |
| `thesis/edge/{id}/weights` | edge → server | C (model weights) |
| `thesis/server/model/global` | server → edge | A/B broadcast, C global model (retained) |

---

## 8. Outputs (where results land)

```
results/strategy_{a,b,c}/run_<timestamp>/
    server.json        rounds[] (accuracy, train time), per_client_bytes_total,
                       bytes_broadcast_total
    client_A.json      per-client upload bytes, timings, label counts
    client_B.json
    server.log, client_*.log, invocation.txt (exact command, for reproducibility)

results/pi_capture/            live-demo outputs
    pi_client_A.json           ticks, detection rate, YOLO ms, upload bytes
    evidence_<ts>/             frame+wav+label triplets + index.html gallery
```

See `docs/EXPERIMENTS.md` for exact commands and `docs/RESULTS.md` for measured
numbers.
