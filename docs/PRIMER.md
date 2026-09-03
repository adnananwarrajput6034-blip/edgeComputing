# Primer — a from-zero walkthrough

A step-by-step explanation of this thesis for someone new to AI. It starts at
"what is AI", builds every concept the project needs, then walks through what we
did, why the simulation is valid, how the physical version works, what the
results prove, and the fusion extension planned for later.

Every **bold term** is defined on first use and collected in the Glossary
(§32). For depth on any topic, the four companion docs are referenced inline:
[ARCHITECTURE](ARCHITECTURE.md), [CODE_FLOW](CODE_FLOW.md),
[RESULTS](RESULTS.md), [EXPERIMENTS](EXPERIMENTS.md).

---

# Part 0 — Orientation

## 1. Who this is for
You know software, but AI is new. This doc closes that gap using *this*
project — no generic tutorial. Read top to bottom once; after that the other
docs will make sense.

## 2. The 30-second summary
Cameras fail in fog and darkness. **Sound** doesn't. We train a small AI model
to recognize objects by the sound they make. To avoid hand-labeling thousands
of sounds, a camera + an off-the-shelf object detector auto-label them. The
thesis question: **where should this model be trained** — on a central server,
partly on the device, or fully on the device? We built all three, measured the
trade-offs, and found the fully-on-device (federated) version uses **16× less
network** while matching accuracy — if one setting is tuned right.

---

# Part 1 — The problem

## 3. The real-world problem
A robot navigates with a camera. In fog, smoke, or darkness the camera is
useless. But a car still *sounds* like a car, a dog still barks. If the robot
could recognize objects by sound, it keeps perceiving when vision fails. That's
the motivation: **audio as a backup sense.**

## 4. Why it's hard
- Recognizing sounds needs a trained model, and training needs **labeled
  examples** ("this sound = car"). Hand-labeling thousands of clips is
  infeasible.
- Edge devices (small computers like a Raspberry Pi) are weak and on slow/
  costly networks. *Where* you do the heavy work (sending data, training)
  matters a lot. That "where" is the thesis.

---

# Part 2 — AI crash course

## 5. AI vs Machine Learning vs Deep Learning
- **AI** — the broad idea of machines doing things that seem intelligent.
- **Machine Learning (ML)** — a way to do AI by *learning patterns from data*
  instead of hand-writing rules.
- **Deep Learning** — ML using **neural networks** with many layers. That's
  what we use.

## 6. What a "model" is
A **model** is just a function: input → output. Ours takes a sound and outputs a
guess like "car (81%)". What makes it AI: the function's behavior is **learned
from examples**, not programmed by hand. The learned settings inside are called
**weights** (also "parameters"). Our model has ~111,000 of them.

## 7. What "training" means (conceptual)
**Training** = adjusting the weights so the model's guesses match the known
answers on example data. Loosely: show it a sound whose true label you know,
compare its guess to the truth, nudge the weights to be a little less wrong,
repeat over many examples. The "how wrong" measure is the **loss**; the nudging
procedure is **gradient descent**. We keep these conceptual on purpose — you
don't need the math to follow the thesis. What matters: *training consumes
labeled examples and produces better weights.*

## 8. Training vs inference
- **Training** — the learning phase (adjust weights). Expensive.
- **Inference** — using the trained model to make a prediction. Cheap.
Different strategies put *training* in different places; inference isn't the
question here.

## 9. Supervised vs unsupervised vs weak supervision — and our choice
- **Supervised learning** — train on examples with known labels (sound + "car").
  Needs labels.
- **Unsupervised learning** — find structure in data with *no* labels.
- **Weak supervision** — labels exist but are *cheap and noisy* rather than
  hand-verified.

**We use weak supervision.** We can't hand-label sounds, so a camera + object
detector produce the labels automatically. They're ~80% correct (not perfect) —
"weak" — but good enough (§16). This choice is the backbone of the whole system.

## 10. Neural networks & CNNs
A **neural network** is layers of simple math units that together learn complex
input→output mappings. A **CNN (Convolutional Neural Network)** is the kind
built for *images* — it detects local patterns (edges, textures) and combines
them. We use a CNN, which is why we first turn each sound into an image (next
part).

---

# Part 3 — Data & representation

## 11. Sound → picture (mel-spectrogram)
A raw sound is a 1-D wiggle over time — bad input for a CNN. We convert each
0.5-second clip into a **spectrogram**: a 2-D image where the x-axis is time,
the y-axis is pitch/frequency, and brightness is energy. A siren *looks like* a
rising-falling stripe. The **mel** scale spaces the frequency axis the way human
hearing works. Exact settings are in Part 8A. Output: a `(51 × 128)` image.
Code: `src/edge/processing/stft.py`.

## 12. The datasets
- **UrbanSound8K** — 8,732 real city sounds (car horn, dog bark, siren, …) in
  **10 classes**, each with a human label. Our main dataset.
- **VGGSound** — used once to *check* how good the auto-labeler is (§16).
- **COCO** — the image dataset YOLO was originally trained on (not ours; YOLO
  comes pre-trained on it).

## 13. Train / validation / test, overfitting, accuracy
To trust a model, you test it on data it never trained on. UrbanSound8K is
pre-split into 10 **folds**; we train on folds 1–8, tune on fold 9
(**validation**), and report on fold 10 (**test** — held out, never seen).
**Overfitting** = memorizing training data instead of learning the general
pattern (looks great in training, fails on the test set); the held-out test
guards against fooling ourselves. **Accuracy** = fraction of test clips
classified correctly. Our baseline model: **70.97%**.

---

# Part 4 — The two models & the labeling trick

## 14. Teacher and student
Two separate models, never confused:
- **YOLO** — a pre-trained *vision* object detector. It's the **teacher/
  labeler**: it looks at the camera frame and says "car". We never train it; we
  just use its output as the label. It never ships anywhere.
- **AudioCNN** — the **student**: it learns sound → object class. *This* is what
  the thesis trains. ~111K weights.

## 15. Cross-modal weak supervision (the core trick)
Camera and mic record the same moment. YOLO names what it *sees* ("car"); we
attach that word to what we *heard*. Vision (a solved problem — good detectors
exist) labels audio (the unsolved thing we're training) — **for free, with no
human**. "Cross-modal" = one sense labels another.

## 16. Validating the trick
Before trusting YOLO's labels, we measured them: on 150 VGGSound clips (which
have human sound-labels), YOLO's frame-label matched ~**80%** of the time. So
~20% of auto-labels are wrong. ML tolerates this: with enough data, the 80%
correct signal outweighs the noise. That 80% is the license to proceed.

---

# Part 5 — The thesis question: where to train?

## 17. Edge vs server
- **Edge device** — the small computer at the sensor (our Raspberry Pi).
- **Server** — a powerful central machine (our laptop / cloud).
Doing work on the edge saves bandwidth and keeps data private but costs edge
battery/CPU. Doing it on the server is easier but sends data over the network.
The thesis measures this trade-off.

## 18. The three strategies (as a story)
Same model, same data — only *where training happens* changes:
- **A — Centralized:** the Pi is a dumb recorder. It ships **raw audio** to the
  server; the server does everything (make spectrograms + train).
- **B — Hybrid:** the Pi does a bit — turns audio into spectrograms — and ships
  those. The server just trains.
- **C — Federated:** the Pi does it all — makes spectrograms **and trains a
  local copy** — and ships only the **trained weights**. The server just
  **averages** everyone's weights. Raw data never leaves the device.

## 19. What each trades
Moving work from server (A) toward edge (C): **network use drops, privacy
improves, server load drops** — but **edge compute rises**. Exact numbers in
§29.

## 20. Federated learning & FedAvg
**Federated learning** = devices train locally and share only model updates, not
data. The server combines updates with **FedAvg (Federated Averaging)**: it
takes a weighted average of the devices' weights. Why ship weights not data?
Weights are (a) small and *constant-size* regardless of how much audio was
recorded, and (b) not the raw audio — so it's private. The formula is in Part 8A.

---

# Part 6 — What we did

## 21. Bootstrap: verify each piece first
Before wiring anything together we proved each part alone: YOLO labels correctly
(~80%), audio→spectrogram works, the AudioCNN can actually learn (70.97% on
clean labels). Testing parts in isolation means that if the full system
misbehaves later, we know which part to blame.

## 22. The harness
To compare strategies we run three programs that talk over a real message bus:
- a **broker** (mosquitto) — a "post office" that routes messages;
- a **server** program — trains (A/B) or averages (C);
- two **client** programs — the two edge nodes.
They communicate over **MQTT** (a lightweight messaging protocol common in
IoT). Details: [CODE_FLOW.md](CODE_FLOW.md).

## 23. Step by step, when you run one strategy
Running `scripts/run_strategy.py --strategy c`:
1. It starts the server, waits until the server is listening.
2. It starts two clients (the fake edge nodes).
3. Clients read UrbanSound8K clips (standing in for a live mic), process them,
   and — depending on the strategy — send raw audio / spectrograms / trained
   weights to the server over MQTT.
4. The server trains or averages, then evaluates accuracy on the held-out test
   set, round by round.
5. Everything is logged to `results/strategy_c/run_*/` (accuracy, bytes sent,
   timings). Commands: [EXPERIMENTS.md](EXPERIMENTS.md).

---

# Part 7 — Simulation vs physical

## 24. What "simulation" means here
We ran the comparison on the laptop, not on real Pis. But **only the sensor
input is simulated** — everything else is real:

| Part | Real or simulated? |
|---|---|
| The two edge "nodes" | simulated (laptop processes, not Pis) |
| Where audio comes from | simulated (recorded UrbanSound8K, not a live mic) |
| The labels | simulated (dataset labels stand in for YOLO's) |
| Broker, MQTT messages, bytes sent | **real** |
| STFT, training, FedAvg | **real** |
| The model and its accuracy | **real** (evaluated on held-out data) |

So: the *inputs* are replayed recordings; the *learning and all measurements are
real*.

## 25. Why simulation is scientifically reasonable
The thesis compares *where training happens* — which affects bandwidth,
compute, privacy. None of that depends on whether audio came from a live mic or
a file. Replaying recorded data is actually *more* rigorous: A, B, and C all
process the **identical** sounds with the **identical** labels, so any
difference is caused by the strategy, not by random room noise. This is a
**controlled experiment** (change one thing, hold the rest fixed).

## 26. The takeaway on simulation
"Simulated" here means *replayed inputs on laptop processes*, not *fake
results*. The training, the network traffic, and the accuracy are all real, so
the comparison is trustworthy. The concrete settings and formulas behind these
runs are collected in Part 8A.

## 27. How the physical version works
Two tracks, **same code**:
- **Science track** (above): laptop + recorded data → the measured comparison.
- **Hardware track**: a real Raspberry Pi 5 with a USB webcam+mic. The **only**
  code change is the *sensor front-end* — instead of reading a file, the Pi
  records 500 ms from the mic and grabs a frame; instead of a dataset label,
  YOLO labels the live frame. Everything after (buffer, send, server) is
  identical, so the server can't tell a real Pi from a simulated one.

We validated this live: a Pi captured for 2 minutes, YOLO labeled what it saw
(person, phone, …), and batches flowed to the laptop server — with an evidence
gallery of frame + sound + label per detection. Only Strategy A has a Pi client
so far; B/C would follow the same pattern.

---

# Part 8A — The math & numbers we actually used

Kept concrete but shallow — the settings and formulas that define our system,
not derivations.

**Audio → spectrogram (STFT):**
- sample rate 16,000 Hz; clip length 0.5 s (8,000 samples)
- analysis window 25 ms (400 samples), step 10 ms (160 samples)
- FFT size 512; **128** mel frequency bins
- result: a `(51 × 128)` image per clip

**Rounds and epochs (high level):**
- an **epoch** = one full pass of the model over its training data.
- a **round** = one cycle of the experiment: in A/B the server trains then
  broadcasts; in C the clients each train locally then the server averages.
- We ran **10 rounds**; in C each client did **1 local epoch** per round.

**FedAvg (Strategy C aggregation):** the new global weights are the
sample-weighted average of the clients' weights:

```
w_global = Σ_k ( n_k / n ) · w_k
```
where `w_k` = weights from client k, `n_k` = number of samples client k trained
on, `n` = total samples. A client that trained on more data counts more.
Code: `src/server/aggregation/fedavg.py`.

**Learning rate (the key tuning knob in C):** the **learning rate** controls how
big a step training takes when nudging weights — too big overshoots, too small
barely moves. Centralized training used `1e-3` (0.001). In C, that same rate
made the averaged model **unstable**; lowering it fixed it, monotonically:

| local learning rate | resulting accuracy | behavior |
|---|---|---|
| 1e-3 | ~0.58 | unstable, oscillates |
| 5e-4 | ~0.65 | still wobbly |
| 1e-4 | ~0.70 | stable, near baseline |
| **5e-5** | **~0.72** | **stable = centralized** |

We chose **5e-5** for C.

**Bandwidth arithmetic (why C wins):**
- A ships raw audio ≈ 12–13 KB per sample → ~66 MB per node for 5,000 samples.
- C ships **weights** ≈ 400 KB per round, **independent of how much audio was
  captured** → ~4 MB per node over 10 rounds. Constant-size updates are the
  whole reason federated saves bandwidth.

**Accuracy** = (correctly classified test clips) / (total test clips), measured
on the never-seen fold-10 held-out set.

---

# Part 8B — Results & what they prove

## 28. The comparison, in plain language
| | A Centralized | B Hybrid | C Federated (5e-5) |
|---|---|---|---|
| Upload per node | 65.9 MB | 48.7 MB | **3.99 MB** |
| Accuracy | 0.71–0.73 | 0.70–0.73 | 0.70–0.74 |
| Edge compute | none | 9.6 s | 158 s |
| Server compute | heavy (trains) | heavy | tiny (averages, 0.5 s/round) |
| Data leaving device | raw audio | spectrograms | only weights |

Read it as: **as we move training from server (A) to device (C), network use
collapses (66 → 4 MB), the server barely works, and privacy improves — at the
cost of the device doing more computing.** Accuracy stays about the same.

## 29. The findings
- **B saves 26%** network vs A — real, but smaller than the ~66% a back-of-
  envelope estimate suggested (raw audio and spectrograms compress similarly).
  *Measuring beat guessing.*
- **C saves 16×** (66 → 4 MB) and keeps data on the device, **matching**
  centralized accuracy — **but only after tuning the learning rate down**
  (§8A). Naïve federated learning was unstable; that instability, and its fix,
  is a genuine result.

## 30. Honest limitations (what we haven't proven yet)
Single-fold test (not full cross-validation); the device split is IID so genuine
non-IID "client drift" is still a separate planned experiment; results are
audio-only (fusion deferred, §31); one webcam's mic was defective and replaced.
Full list: [RESULTS.md](RESULTS.md) §8.

---

# Part 9 — The extension: fusion model

> **Update:** this extension has since been BUILT and measured — the full
> story (data harvesting, a instructive failure, the fix, and the results
> proving "audio helps when vision fails") is in **[FUSION.md](FUSION.md)**,
> written as the sequel to this primer. The section below is the original
> plan, kept for context.

## 31. What it is, why deferred, what it will show
So far the model uses **audio only**. The **fusion model** combines *two*
senses at once: a vision branch (a light image model) **and** the audio branch,
merged to make one prediction. Why add it: to directly demonstrate the thesis's
motivating claim — **"audio helps when vision fails."** With both inputs, we can
show the model still recognizes a car in darkness (vision blank, audio strong),
and that fusion beats either sense alone.

Why it's deferred: training a joint audio+vision model needs **paired**
audio+video examples (same moment, both sensors) with labels — a dataset we
don't yet have. Building/collecting that is real work, and it adds a second
variable to every experiment. So we deliberately finished the cleaner
audio-only A/B/C comparison first. The fusion model is already *built and
wired for a forward pass* (`src/server/training/models/fusion_model.py`); what
remains is data + training. Planned as a later chapter/commit to show the
vision-fails-audio-saves result.

Note: even today the system is "audio-visual" in one sense — **vision labels
audio** (§15). The fusion model would make it audio-visual at *inference* too.

---

# Part 10 — Reference

## 32. Glossary
- **AI / ML / Deep Learning** — intelligent behavior / learning from data /
  learning with deep neural networks.
- **Model** — a learned input→output function; its learned settings are
  **weights/parameters** (~111K here).
- **Training / inference** — the learning phase / the using phase.
- **Loss** — a measure of how wrong the model's guesses are (kept conceptual).
- **Learning rate** — how big a step training takes; a key tuning knob (we use
  5e-5 for C).
- **Supervised / unsupervised / weak supervision** — learn with labels / without
  labels / with cheap noisy labels (we use weak supervision).
- **CNN** — a neural network for image-like inputs (we feed it spectrograms).
- **Spectrogram / mel** — sound turned into a `(51×128)` image; mel = human-
  hearing frequency scale.
- **Label** — the correct answer for a training example ("car"); ours come from
  YOLO.
- **Epoch / round** — one pass over the data / one experiment cycle
  (train-or-average + evaluate).
- **Overfitting** — memorizing training data; caught by the held-out test set.
- **Accuracy** — fraction of held-out clips classified correctly.
- **Edge / server** — the small sensor device / the powerful central machine.
- **Federated learning / FedAvg** — devices train locally and share weights; the
  server averages them, `w_global = Σ (n_k/n)·w_k`.
- **MQTT / broker** — a lightweight messaging protocol / its routing server.
- **YOLO** — the pre-trained vision detector we use as the auto-labeler.
- **IID / non-IID** — data split evenly across devices / unevenly (skewed).
- **Fusion model** — a model using audio + vision together (deferred).

## 33. How the docs fit together
- **PRIMER.md** (this) — learn the whole thing from zero.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the system design reference.
- **[CODE_FLOW.md](CODE_FLOW.md)** — how the code executes, file by file.
- **[RESULTS.md](RESULTS.md)** — full measured numbers and findings.
- **[EXPERIMENTS.md](EXPERIMENTS.md)** — exact commands to reproduce.
