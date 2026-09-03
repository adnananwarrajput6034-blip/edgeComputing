# System Architecture

The single design reference for the project. For *how the code executes* see
[CODE_FLOW.md](CODE_FLOW.md); for *measured numbers* see RESULTS.md.

---

## 1. Problem & approach

Goal: let an edge device recognize objects **by their sound**, so perception
survives when the camera fails (fog, darkness). The research question is not
"can it hear?" but **where should the model be trained** — on a server, partly
on-device, or fully on-device — and what that costs in bandwidth, compute,
privacy, and accuracy.

**Two models, never confused:**

| Model | Role | Trained here? | Ships? |
|---|---|---|---|
| **YOLOv8** (frozen) | Labeler — detects objects in the camera frame; its label is attached to the simultaneous audio | No (pre-trained COCO) | Never — runs on edge, output used then discarded |
| **AudioCNN** | Student — learns mel-spectrogram → object class | **Yes** — this is what the strategies train | Weights only, and only in Strategy C |

This is **cross-modal weak supervision**: vision (a solved problem) labels
audio (the thing we train) for free, with no human annotation. Label noise was
measured at ~20% (YOLO ~80% precision on VGGSound), which weak supervision
tolerates.

> The audio+vision **FusionModel** (a joint model using both modalities at
> inference) is designed but **deferred** — it needs paired audio+video data we
> don't yet have. All current results are audio-only. "Fusion" today lives at
> the *supervision* level (vision labels audio), not in a joint trained model.

---

## 2. Two tracks

The project runs on two tracks that share the same code:

| | Science track | Hardware track |
|---|---|---|
| Runs on | Laptop | Raspberry Pi 5 + laptop server |
| Data | UrbanSound8K replayed over real MQTT | Live webcam + USB mic capture |
| Labels | Dataset ground truth (stands in for YOLO) | YOLO on live frames |
| Purpose | The measured A/B/C comparison (needs identical data + ground-truth labels) | Validate the pipeline on real edge hardware; the demo |
| Status | A, B, C complete | Strategy A validated end-to-end on one Pi |

Only the **sensor front-end** differs between tracks; buffering, serialization,
MQTT transport, and the server are identical (see CODE_FLOW.md §5).

---

## 3. Layered architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  SERVER LAYER (laptop / cloud)                                     │
│    A/B: receive data → (STFT) → model.fit → broadcast weights     │
│    C:   receive weights → FedAvg average → broadcast global model  │
│    Model repository (global AudioCNN weights)                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                        MQTT (mosquitto)
              localhost (science)  /  LAN 0.0.0.0 (Pi)
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                            ▼
┌───────────────────────┐                ┌───────────────────────┐
│  EDGE NODE A           │                │  EDGE NODE B           │
│  (fake proc / Pi 5)    │                │  (fake proc / Pi 5)    │
│   camera → YOLO label  │                │   camera → YOLO label  │
│   mic → audio          │                │   mic → audio          │
│   [B/C: STFT]          │                │   [B/C: STFT]          │
│   [C: local training]  │                │   [C: local training]  │
│   buffer → publish     │                │   buffer → publish     │
└───────────────────────┘                └───────────────────────┘
```

Two edge nodes is the thesis scope; the architecture scales to N.

---

## 4. The three strategies (data flow)

Same model, same data — only **where STFT and training run and what crosses the
wire** changes.

```
A — CENTRALIZED
  edge:   capture → YOLO label → buffer RAW AUDIO → publish
  server: STFT → model.fit → broadcast weights
  wire:   raw audio + labels        (most bandwidth, least privacy)

B — HYBRID
  edge:   capture → YOLO label → STFT → buffer SPECTROGRAM → publish
  server: model.fit → broadcast weights
  wire:   spectrograms + labels     (edge does STFT)

C — FEDERATED
  edge:   capture → YOLO label → STFT → LOCAL model.fit → publish WEIGHTS
  server: FedAvg average → broadcast global model
  wire:   model weights only        (least bandwidth, data never leaves device)
```

As training moves toward the edge (A→B→C): bandwidth and server load drop,
privacy improves, edge compute rises. Measured trade-offs in RESULTS.md.

---

## 5. Audio pipeline

Raw waveform → mel-spectrogram "image" for the CNN
(`src/edge/processing/stft.py`):

| Parameter | Value |
|---|---|
| Sample rate | 16 000 Hz |
| Clip window | 500 ms (8000 samples) |
| STFT window / hop | 400 / 160 samples (25 ms / 10 ms) |
| FFT size | 512 |
| Mel bins | 128 |
| Output shape | `(51, 128)` → `(51, 128, 1)` with channel |

Mel scale (vs linear Hz) matches human hearing — more resolution where audio
information lives. Standard audio-ML recipe.

---

## 6. Models

**AudioCNN** (`src/server/training/models/audio_cnn.py`) — the student:

```
Input (51,128,1)
  3× [Conv2D → BatchNorm → ReLU → MaxPool]   (32 → 64 → 128 filters)
  GlobalAveragePooling2D
  Dense(128) → Dropout
  Dense(10, softmax)
Total ≈ 111K parameters   (small enough to train on a Pi)
Baseline: 70.97% test accuracy on UrbanSound8K fold-10 holdout
```

**YOLOv8n** (`src/edge/processing/vision.py`, `VisionProcessor`) — frozen
labeler. ~47 ms/frame on a Mac, ~430 ms/frame on the Pi 5 CPU. Outputs a class
name + confidence; the frame is discarded after.

**FusionModel** (`src/server/training/models/fusion_model.py`) — trained on
VGGSound pairs: 0.847 clean / 0.551 blackout (recipe: audio warm-start +
50% modality dropout — see RESULTS.md §8). MobileNetV3-Small (frozen) +
AudioCNN branch + fusion head, ~1.33M params. **FVFusionModel** is its
wire-format twin for strategies: vision input = the frozen backbone's
576-float feature vector (computed on the edge; frames never ship), proven
prediction-identical to the image model.

---

## 7. Communication

**Transport:** MQTT via mosquitto. Science runs use `configs/mosquitto.conf`
(localhost); Pi runs use `configs/mosquitto_lan.conf` (0.0.0.0, LAN-reachable).
32 MB packet limit for multi-MB raw-audio batches.

**Topics:**

| Topic | Direction | Used by |
|---|---|---|
| `thesis/edge/{id}/data` | edge → server | A, B (raw audio / spectrograms) |
| `thesis/edge/{id}/weights` | edge → server | C (model weights) |
| `thesis/server/model/global` | server → edge | A/B broadcast; C global model (retained) |

**Federated aggregation:** hand-rolled **FedAvg over MQTT**
(`src/server/aggregation/fedavg.py`), *not* Flower. Rationale: keeping the
transport identical across A/B/C makes bandwidth directly comparable — Flower's
gRPC would introduce a confound. Flower is noted as the production-grade path;
the Flower scaffolds were removed in favor of the MQTT implementation.

**Serialization:** `np.savez_compressed` (+ `zlib` for A/B data batches). Byte
counts are recorded independently by sender and receiver and cross-checked.

---

## 8. Hardware (actual)

| Device | Spec | Notes |
|---|---|---|
| Edge × 2 | Raspberry Pi 5, 8 GB, Debian 13 | uni-provisioned; Python 3.13 |
| Camera + mic | LogiLink USB webcam (UVC + USB audio) | one unit = both sensors, plug-and-play |
| Server | Laptop (macOS) | runs broker + server + science sims |

> The webcam's built-in mic on one unit was defective (constant full-scale
> output); a separate USB mic is used for audio. A real edge-deployment lesson:
> validate sensors by inspecting the data, not by trusting the driver.

This supersedes earlier design assumptions (Pi 4, Raspberry Pi OS, Camera
Module 3 + I2S mic) — none of which match the delivered hardware.

---

## 9. Key design decisions

- **Strategy pattern / shared spine** — one pipeline, swappable sample source
  and swappable strategy. Lets the simulation *be* the real system with a
  different front-end (fake-Pi ↔ real-Pi is a two-line change).
- **Producer-consumer on the server** — the MQTT callback only enqueues; the
  main thread does STFT/training/aggregation, so long compute never blocks the
  network thread.
- **Warm-start** — strategies continue-train a pre-trained AudioCNN (the
  realistic deployment case), not train from scratch.
- **Hand-rolled FedAvg over MQTT** — comparability over framework convenience.
- **Fresh local optimizer per round (C)** — FedAvg-correct; carrying Adam state
  across `set_weights()` corrupts updates.

---

## 10. Status & roadmap

**Done:** Strategies A, B, C implemented and measured (science track); Strategy
A validated live on one Pi.

**Next:**
1. Model-freshness experiment — inject unseen classes mid-stream, measure
   time-to-adapt per strategy (the thesis centerpiece).
2. Non-IID experiment — skewed class split across nodes; FedProx.
3. Pi-hardware validation of C (real federated training across two Pis).
4. Fusion track: F1–F6 done (paired data, trained fusion model, blackout
   result, FB/FC strategies) — F7 Pi feasibility timing remains.

Repository layout and per-file roles: see [README.md](../README.md).
Code execution: [CODE_FLOW.md](CODE_FLOW.md).
