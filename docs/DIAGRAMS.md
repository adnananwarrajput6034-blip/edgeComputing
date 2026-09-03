# Diagrams

Mermaid source for the report figures. GitHub renders these inline; for the
thesis document, export any diagram to SVG/PNG at https://mermaid.live (paste
the block) or with `mmdc` (mermaid-cli).

Contents:
1. System architecture — current (AudioCNN)
2. System architecture — fusion era (planned)
3. Sequence: Strategies A/B (data upload → server training)
4. Sequence: Strategy C (federated rounds, FedAvg)
5. Sequence: live Pi capture (hardware track)

---

## 1. System architecture — current (AudioCNN)

The system as measured: two edge nodes, MQTT broker, one server. The three
strategies differ only in where STFT/training run and what crosses the wire.

```mermaid
---
title: "System architecture (current) — AudioCNN, strategies A/B/C"
---
flowchart TB
    subgraph EDGE_A["Edge node A (Pi 5 / simulated)"]
        camA["Camera frame"] --> yoloA["YOLOv8 (frozen)<br/>label = 'car'"]
        micA["Mic 500 ms audio"] --> prepA["Strategy prep<br/>A: raw audio<br/>B: STFT to spectrogram<br/>C: STFT + local training"]
        yoloA -- "label" --> prepA
        prepA --> bufA["Buffer (batch)"]
    end

    subgraph EDGE_B["Edge node B (Pi 5 / simulated)"]
        camB["Camera frame"] --> yoloB["YOLOv8 (frozen)"]
        micB["Mic 500 ms audio"] --> prepB["Strategy prep"]
        yoloB -- "label" --> prepB
        prepB --> bufB["Buffer (batch)"]
    end

    subgraph NET["Network"]
        broker[("MQTT broker<br/>(mosquitto)")]
    end

    subgraph SERVER["Server (laptop / cloud)"]
        ingest["Ingest queue"] --> role["A: STFT + train<br/>B: train<br/>C: FedAvg average"]
        role --> cnn["AudioCNN (111K params)<br/>global model"]
        cnn --> eval["Evaluate on held-out test set"]
        cnn --> bcast["Broadcast weights"]
    end

    bufA -- "A/B: data + labels<br/>C: model weights" --> broker
    bufB -- "A/B: data + labels<br/>C: model weights" --> broker
    broker --> ingest
    bcast --> broker
    broker -. "global model" .-> prepA
    broker -. "global model" .-> prepB
```

---

## 2. System architecture — fusion era (planned, F-track)

The FusionModel adds a vision branch at inference. Frames still never leave
the device: strategies A/B ship the MobileNetV3 **feature vector** (256
numbers, not reversible to an image) alongside the audio; C is unchanged
(everything stays local).

```mermaid
---
title: "System architecture (planned) — FusionModel, frames never leave the edge"
---
flowchart TB
    subgraph EDGE["Edge node (Pi 5)"]
        cam["Camera frame"] --> yolo["YOLOv8 (frozen)<br/>label"]
        cam --> mnet["MobileNetV3-Small (frozen)<br/>256-dim feature vector"]
        mic["Mic 500 ms audio"] --> stft["STFT<br/>(51x128) spectrogram"]
        yolo --> pack["Sample: (vision FV, spectrogram, label)"]
        mnet --> pack
        stft --> pack
        pack --> strat["A/B: upload sample<br/>C: local fusion training"]
        cam -. "frame discarded<br/>after YOLO + MobileNet" .-> discard[("never ships")]
    end

    strat --> broker[("MQTT broker")]

    subgraph SERVER["Server"]
        broker --> train["A/B: train FusionModel<br/>C: FedAvg average"]
        subgraph FUSION["FusionModel (~1.33M params)"]
            vin["vision FV 256"] --> concat["concat 384"]
            ain["audio branch 128<br/>(AudioCNN)"] --> concat
            concat --> head["fusion head<br/>Dense 256 - 128 - softmax"]
        end
        train --> FUSION
        FUSION --> bcast2["broadcast weights"] --> broker
    end
```

---

## 3. Sequence — Strategies A/B (data upload, server training)

One full cycle. The only A-vs-B difference is *who* runs STFT (highlighted).

```mermaid
---
title: "Strategies A/B — edge uploads data, server trains"
---
sequenceDiagram
    autonumber
    participant CA as Client A (edge)
    participant CB as Client B (edge)
    participant BR as MQTT broker
    participant SV as Server

    Note over CA,SV: A ships RAW AUDIO (server does STFT).<br/>B ships SPECTROGRAMS (edge does STFT).<br/>Everything else is identical.
    Note over SV: warm-start AudioCNN,<br/>subscribe thesis/edge/+/data
    SV->>SV: print "Server ready"

    loop every 500 ms (sensor cadence)
        CA->>CA: capture audio + label<br/>(B: + STFT on edge)
        CA->>CA: buffer sample
    end

    CA->>BR: publish batch (500 samples,<br/>A: raw audio / B: spectrograms, zlib)
    BR->>SV: deliver thesis/edge/A/data
    CB->>BR: publish batch
    BR->>SV: deliver thesis/edge/B/data

    SV->>SV: decompress<br/>(A only: STFT on server)
    SV->>SV: add to training pool

    alt pool reaches trigger (e.g. 1000)
        SV->>SV: model.fit on full pool (1 epoch)
        SV->>SV: evaluate on held-out fold-10
        SV->>BR: broadcast weights<br/>thesis/server/model/global
        BR--)CA: updated model (optional)
        BR--)CB: updated model (optional)
        Note over CA,CB: In A/B the edge does NOT train, so it does not need weights for learning. The broadcast exists because (1) a deployed edge uses the updated model for ON-DEVICE INFERENCE, and (2) it is counted as download traffic so A/B/C bandwidth is compared on equal terms. The simulation clients do not consume it.
    end

    Note over CA,SV: repeats until num_rounds reached
```

---

## 4. Sequence — Strategy C (federated rounds, FedAvg)

Round-synchronized: the server never trains, only averages. Data never leaves
the clients; only weights travel (both directions).

```mermaid
---
title: "Strategy C — federated rounds: clients train, server averages (FedAvg)"
---
sequenceDiagram
    autonumber
    participant CA as Client A (edge)
    participant CB as Client B (edge)
    participant BR as MQTT broker
    participant SV as Server

    Note over CA,SV: Data NEVER leaves the clients — only model weights travel, in both directions.
    Note over CA,CB: build LOCAL dataset once<br/>(own audio -> STFT, stays on device)
    Note over SV: warm-start global model,<br/>subscribe thesis/edge/+/weights

    SV->>BR: broadcast global model (round 1,<br/>RETAINED + local_epochs + local_lr)
    BR-->>CA: global model r1
    BR-->>CB: global model r1

    par local training (in parallel)
        CA->>CA: set_weights(global)<br/>fresh Adam(local_lr)<br/>fit(local data, 1 epoch)
        CB->>CB: set_weights(global)<br/>fresh Adam(local_lr)<br/>fit(local data, 1 epoch)
    end

    CA->>BR: publish weights + n_samples (round-tagged)
    BR->>SV: weights from A
    CB->>BR: publish weights + n_samples
    BR->>SV: weights from B

    SV->>SV: FedAvg: w_global = sum(n_k/n * w_k)
    SV->>SV: evaluate on held-out fold-10
    SV->>BR: broadcast global model (round 2)

    Note over CA,SV: repeat for num_rounds,<br/>then broadcast done=true
    BR-->>CA: done -> client exits
    BR-->>CB: done -> client exits
```

---

## 5. Sequence — live Pi capture (hardware track, Strategy A validated)

The real-hardware demo: same downstream pipeline as the simulation; only the
sample source differs. Frames are discarded on-device.

```mermaid
---
title: "Live Pi capture (hardware track) — frames are discarded on-device"
---
sequenceDiagram
    autonumber
    participant HW as Webcam and USB mic
    participant PI as Pi client (strategy_a_pi_client)
    participant BR as MQTT broker (laptop)
    participant SV as Server (laptop)

    Note over PI: startup: open camera + mic,<br/>load YOLO (+1 warmup inference)

    loop every ~1 s tick
        PI->>HW: start 500 ms mic recording
        PI->>HW: grab frame (mid-recording)
        HW-->>PI: audio + frame
        PI->>PI: YOLO on frame (~430 ms on Pi 5)
        alt object detected (confidence 0.5 or higher)
            PI->>PI: sample = (audio, label, confidence)<br/>-> buffer. FRAME DISCARDED
        else nothing detected
            PI->>PI: tick skipped<br/>(no label -> no sample)
        end
    end

    alt buffer full (25 samples)
        PI->>BR: publish compressed batch<br/>(audio + labels only)
        BR->>SV: deliver batch
        SV->>SV: count bytes, unpack,<br/>label-check vs class list
    end

    Note over PI,SV: byte counts match sender and receiver.<br/>Images never crossed the network
```

---

### Exported figures (for the report)

Rendered PNGs live in [figures/](figures/):

| # | File | Diagram |
|---|---|---|
| 1 | `figures/01_architecture_current_audiocnn.png` | §1 architecture (current) |
| 2 | `figures/02_architecture_fusion_planned.png` | §2 architecture (fusion era) |
| 3 | `figures/03_sequence_strategies_ab.png` | §3 sequence A/B |
| 4 | `figures/04_sequence_strategy_c_fedavg.png` | §4 sequence C (FedAvg) |
| 5 | `figures/05_sequence_live_pi_capture.png` | §5 sequence live Pi |

Re-export after editing a diagram so source and PNG stay in sync.

### Regenerating / exporting

- GitHub renders these blocks natively.
- For the thesis: paste a block into https://mermaid.live → export SVG/PNG,
  or `npx -y @mermaid-js/mermaid-cli -i docs/DIAGRAMS.md -o figures/` to batch-export.
- Keep diagrams in sync with [ARCHITECTURE.md](ARCHITECTURE.md) (design) and
  [CODE_FLOW.md](CODE_FLOW.md) (execution) when the system changes.
