# Results

Measured outcomes of the strategy comparison. All science-track runs: laptop,
UrbanSound8K replayed over a real MQTT broker, warm-started AudioCNN (baseline
70.97% on fold-10), 2 edge nodes, 10 rounds, ~IID class split.

Reproduce: see [EXPERIMENTS.md](EXPERIMENTS.md). Design context:
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Headline comparison (A vs B vs C)

| Metric | A Centralized | B Hybrid | C Federated (LR 5e-5) |
|---|---|---|---|
| **Upload / node** | 65.9 MB | 48.7 MB | **3.99 MB** |
| Download (broadcast) | 4.0 MB | 4.0 MB | 4.9 MB |
| Test accuracy | 0.71–0.73 | 0.70–0.73 | 0.70–0.74 |
| Edge compute | 0 | 9.6 s (STFT) | 158 s (local training) |
| Server compute | trains, 3.7→26.3 s/round | trains | **averages only, 0.5 s/round** |
| Data leaving device | raw audio | spectrograms | **only weights** |

**Takeaway:** moving training toward the edge (A→B→C) collapses upload
bandwidth (66 → 49 → 4 MB) and server load, and improves privacy; the cost is
edge compute (0 → 10 → 158 s). Federated matches centralized accuracy — with a
tuning caveat (§4).

---

## 2. Strategy A — Centralized (baseline)

Raw audio → server; server does STFT + training on a growing pool.

| Round | Pool | Test acc | Train time |
|---|---|---|---|
| 1 | 1000 | 0.712 | 3.7 s |
| 5 | 5000 | 0.710 | 14.6 s |
| 10 | 10000 | 0.712 | 26.3 s |

- Upload: **66.0 / 65.9 MB** (node A / B) for 5000 samples each (~12–13 KB/sample compressed).
- **Accuracy flat ~0.71** — expected: the model was already trained on this
  distribution, so more of the same data adds little. Flat = healthy baseline.
- **Train time grows linearly** (3.7 → 26.3 s) because centralized re-trains on
  everything ever received — a real scalability cost.

---

## 3. Strategy B — Hybrid (edge STFT)

Edge does STFT → ships spectrograms; server trains only.

- Upload: **48.6 / 48.7 MB** per node → **26% less than A**.
- Accuracy: **0.70–0.73** — no penalty vs A.
- Edge compute added: **9.65 s** of STFT per node over 5000 samples.

**Honest finding:** the bandwidth win is **~26%, not the ~66%** the project's
theoretical estimate implied. A raw int16 clip (16 KB) and a float16
spectrogram (13 KB) start close, and both compress similarly on the wire — so
the analytical model overstates B's advantage. Measuring mattered.

---

## 4. Strategy C — Federated (FedAvg over MQTT)

Clients train locally; server only averages weights. Upload is **weights per
round, independent of data volume**.

- Upload: **3.99 MB / node** over 10 rounds (~400 KB/round) — **16× less than A.**
- Server compute: **0.5 s/round** (just averaging) vs A's 3.7–26.3 s of training.
- Edge compute: **158 s** local training per node (the cost of moving training
  to the edge).

### The local-learning-rate finding

Naïve FedAvg (local LR 1e-3, the same rate A/B use on the server)
**destabilizes** the warm-started model — even though the data split is IID.
Averaging two models that each took a full epoch of large steps lands *off* the
converged minimum, and it oscillates. Lowering the local LR fixes it
monotonically:

| Local LR | Accuracy trajectory | Behavior |
|---|---|---|
| 1e-3 | 0.51–0.66 (~0.58) | unstable |
| 5e-4 | 0.55–0.72 (~0.65) | still oscillating |
| 1e-4 | 0.68–0.73 (~0.70) | stable, near baseline |
| **5e-5** | **0.70–0.74 (~0.72)** | **stable, = centralized** |

**Interpretation:** a converged model needs *gentler* local updates under
FedAvg than centralized training does, so clients don't diverge before
averaging. At LR 5e-5, C reaches centralized parity (~0.72) at 16× less
bandwidth with data never leaving the device. This instability (and its fix) is
a genuine, citable result — it is *why* careful LR schedules and FedProx exist.

Note this instability was **not** client drift from non-IID data (the split is
IID); genuine label-skew drift is a separate, planned experiment.

---

## 5. Baseline model (AudioCNN)

Trained on UrbanSound8K (folds 1–8 train, 9 val, 10 test), sliding-window
augmentation, class-balanced weights.

- **Test accuracy: 70.97%** (fold-10 holdout), 111K parameters.
- Weakest class: **siren** (~0.59 F1) — fewer samples, acoustically variable.
- Limitation: single-fold split (not full 10-fold CV) — run 10-fold before the
  final thesis report for mean ± std.

---

## 6. Labeler validation (YOLO)

- **~80% precision** as an audio labeler (VGGSound: YOLO on the frame vs the
  human sound label). Low-confidence frames produce no label (skipped, not
  mislabeled) — so the ~20% is the noise the audio model trains under.
- Latency: ~47 ms/frame (Mac), ~430 ms/frame (Pi 5 CPU) — fine for a ≥500 ms
  capture cadence.

---

## 7. Real-hardware validation (Strategy A on Pi 5)

Live 2-minute capture: Raspberry Pi 5 + USB webcam/mic → YOLO labels → MQTT →
laptop server (the **unchanged** science-track server).

- 119 ticks / 2 min, **~26% labeled** (YOLO fired on person / cell phone /
  keyboard / traffic light; the rest correctly skipped).
- **452 KB uploaded**; server byte count matched the client to the byte.
- Evidence gallery (`results/pi_capture/evidence_*/index.html`): per labeled
  tick, the frame YOLO saw + the 500 ms audio + the assigned label. Frames are
  **discarded in the production path** — only audio + label ship.
- A `vase` mislabel appears in the gallery — a live specimen of the ~20% label
  noise.

Proves the pipeline runs unchanged on real edge hardware: the server cannot
tell a real Pi from a simulated one.

---

## 8. Fusion track (F3–F5): audio + vision, and darkness

Trained on 973 VGGSound pairs (5 classes: bird/car/cat/dog/motorcycle;
3,008 train samples, 98-clip test set, chance = 0.20). "Blackout" = every
frame replaced by black (fog/darkness/dead camera); audio unchanged.

| Model | Clean | Blackout | Behavior |
|---|---|---|---|
| vision-only | 0.806 | 0.204 | collapses to chance |
| naive fusion | 0.745 | 0.173 | worse than vision alone; ignores audio |
| fusion + dropout (0.3/0.5) | 0.745 | ~0.21 | dropout alone does NOT fix it |
| **fusion + dropout 0.5 + audio-warmstart** | **0.847** | **0.551** | best clean AND holds the audio floor |
| audio-only (floor) | 0.561 | 0.561 | unaffected — the reference |

**Findings:**
- **Fusion beats either modality alone** (0.847 > 0.806 vision, > 0.561 audio) —
  but only with the right training recipe.
- **"Audio helps when vision fails," measured:** in blackout, vision-only is
  random (0.204) while the fused model keeps 0.551 ≈ the full audio floor.
- **The recipe is the result:** naive fusion ignores audio (modality gradient
  starvation — a from-scratch audio branch cannot compete with a pretrained
  vision branch: its features stay near-constant and the head wires it out).
  Modality dropout alone cannot revive a starved branch. Warm-starting the
  audio branch from the trained audio-only model, then applying 50% modality
  dropout, cures both problems at once.
- Diagnosis method worth noting: identical-accuracy runs flagged a dead
  branch; probes (constant blackout predictions, zero audio-sensitivity,
  near-zero feature variance) located it. Small test set (98 clips) → treat
  point estimates as ±~5%.

Reproduce: `train_fusion.py` (baselines, `--modality-dropout`,
`--audio-warmstart`), `eval_fusion_blackout.py`. Raw numbers:
`data/vggsound/fusion_blackout_results.json`.

### F6 — strategies with the fusion model

The edge ships the frozen backbone's 576-float **feature vector** (1.1 KB,
not reversible to an image) instead of the frame; the server trains an
FV-input fusion model proven equivalent to the image model (100% prediction
agreement). Both runs warm-start from the 0.847/0.551 model and log
blackout accuracy per round:

| | FB (hybrid-fusion) | FC (federated-fusion) |
|---|---|---|
| What ships | spectrogram + FV (10.3 KB/sample) | weights only (1.41 MB/round) |
| Upload/node | 14.78 MB (1,502 samples) | 13.76 MB (10 rounds) |
| Clean acc | dips, recovers to 0.837 | stable 0.84–0.88 |
| Blackout acc | **0.52 → 0.67 over rounds** | **0.59–0.68** |
| Server compute | trains, 2.6→7.3 s/rd | averages, 0.2 s/rd |

Findings: (1) **graceful degradation survives — and improves under —
continued strategy training** (blackout ~0.55 baseline → ~0.67); (2)
**federated cost scales with model size, not data**: 1.41 MB/round vs
audio-C's 400 KB ≈ the 3.5× trainable-parameter ratio (390K vs 111K); at
this data volume FB and FC totals are similar, at 10× data FB grows ~10×
while FC is unchanged; (3) the FV wire format costs ~10 KB/sample vs
~30–50 KB for JPEG frames, with frames never leaving the device.
Reproduce: `run_strategy.py --strategy fb|fc` (needs `prepare_fusion_fv.py`
first).

---

## 9. Limitations & honesty notes

- **B's bandwidth win is 26%, not the estimated ~66%** (§3).
- **Naïve FedAvg destabilizes**; C needs a tuned-down local LR (§4).
- **Single-fold evaluation** (fold-10 holdout), not 10-fold CV — do CV for the
  final report.
- **IID split so far** — non-IID (label-skew) client drift is a separate
  planned experiment; today's C instability is LR-driven, not drift.
- **Audio-only** — the joint audio+vision FusionModel is deferred (needs paired
  data); "fusion" currently means vision-labels-audio supervision.
- **One webcam's built-in mic was defective** (constant full-scale output); a
  separate USB mic is used. A real edge-deployment caveat.

---

## 10. Where the numbers come from

```
results/strategy_a/run_*/server.json     A: rounds[], per_client_bytes_total
results/strategy_b/run_*/server.json     B: + client_A.json stft_time_total
results/strategy_c/run_*/server.json     C: one run per local LR (the sweep)
results/pi_capture/                       live demo + evidence gallery
```

Byte counts are recorded independently by sender (client_*.json) and receiver
(server.json) and cross-checked. Each run dir has `invocation.txt` with the
exact command.
