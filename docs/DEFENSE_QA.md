# Defense Q&A — anticipated questions and answers

Questions a supervisor/examiner is likely to ask, grouped by angle, each
with a short answer, the deeper reasoning, and — where relevant — what we
would do if asked to go further. Numbers come from [RESULTS.md](RESULTS.md);
concepts from [PRIMER.md](PRIMER.md) and [FUSION.md](FUSION.md).

---

## A. The core idea

**A1. Why label audio with a vision model? Isn't that circular?**
Short: vision is a solved problem with free pre-trained detectors; audio
recognition for our classes is not — so the solved sense teaches the
unsolved one.
Deeper: it's not circular because the two models answer different questions
at different times. YOLO answers "what is visible NOW" (needs light); the
audio model learns "what does X sound like" so it can answer LATER without
light. Once trained, the audio model works when YOLO cannot.
If pushed: we validated the assumption empirically — YOLO's frame label
matches the human sound label ~80% of the time (150 VGGSound clips).

**A2. Your labels are 20% wrong. How can training on wrong labels work?**
Short: with enough data, the 80% consistent signal outweighs the 20%
inconsistent noise — this is standard weak supervision.
Deeper: wrong labels are *inconsistent* (a dog clip labeled "car" today,
another labeled "bird" tomorrow) while right labels always agree, so
gradients from correct labels accumulate and noise partially cancels.
Literature: deep networks tolerate substantial label noise (e.g. Rolnick et
al., 2017). We also measured the noise rate BEFORE building on it, so the
risk was quantified, not hoped away.
Honest limit: 80% precision is for detections; YOLO often detects nothing
(low recall) — those samples are *skipped*, costing data volume, not
correctness.

**A3. Why audio at all? Why not radar/lidar for fog?**
Short: audio is passive, ultra-cheap (a microphone), low-power, and
complements vision with zero emissions; radar/lidar are active, costly, and
power-hungry — wrong class of edge device.
Also: the thesis question is really about training placement; audio-object
recognition is the *vehicle*, chosen because it makes the cross-modal
labeling trick possible.

---

## B. Experimental design

**B1. Your comparison ran in simulation. Why should we trust it?**
Short: only the *sensors* are simulated — the training, the MQTT network
traffic, and the measured bytes/accuracy are all real.
Deeper: the thesis compares WHERE training runs. That depends on what is
computed and transmitted, not on whether audio came from a live mic or a
replayed file. Replay is actually *stronger* science here: all strategies
process identical data with identical labels, so differences are caused by
the strategy alone (controlled experiment). Live capture can't guarantee
that.
Plus: we separately validated the pipeline on real hardware (live Pi 5
capture → YOLO → MQTT → the *unchanged* server; byte counts matched
sender/receiver exactly).

**B2. Why UrbanSound8K replay instead of live data for the science runs?**
Short: fair comparison needs identical inputs across strategies, and
accuracy measurement needs ground-truth labels — live capture provides
neither.
Live YOLO labels (person, cell phone…) have no ground truth to score
against; datasets do.

**B3. Why only 2 edge nodes?**
Short: two nodes is the minimum that makes federation meaningful
(aggregation of multiple parties) and matches the thesis's stated hardware
scope; the architecture scales to N unchanged (FedAvg is a weighted mean
over any number of clients).
If pushed: we would simulate N>2 clients as additional processes — the
harness needs only a list extension — and expect the known FL result that
more clients smooth the average but slow per-round wall time.

**B4. Single-fold evaluation, 98-clip fusion test set — is that enough?**
Short honest answer: adequate for *relative* comparisons and trends, not
for precise point estimates; we treat fusion numbers as ±~5% and flag
10-fold cross-validation as pending for the final report.
What we'd do: 10-fold CV on UrbanSound8K (~5 h) for mean ± std; harvest a
larger VGGSound test split for tighter fusion intervals.

**B5. Why is accuracy flat (~0.71) across all strategy rounds? Doesn't that
mean nothing is learned?**
Short: flat is the *expected and correct* result — the model was
pre-trained on the same distribution, so new same-distribution data cannot
add much. The experiment measures the COST of continued training per
placement, not accuracy gains.
The accuracy-motion experiments are separate: the LR sweep (C), the fusion
blackout runs, and the planned model-freshness experiment (inject unseen
classes, measure adaptation speed — where curves genuinely move).

---

## C. Strategy comparison results

**C1. B saved only 26% bandwidth, not the ~66% your own docs estimated.
Why?**
Short: the theoretical estimate ignored compression. Raw int16 audio
(16 KB) and float16 spectrograms (13 KB) start close, and zlib narrows the
gap further.
Why we keep this in the thesis: it's a finding — analytical bandwidth
models overstate preprocessing gains; measurement corrected a published
assumption of our own design. Honesty here strengthens the rest.

**C2. Federated (C) matched centralized accuracy. Is that not suspicious?**
Short: it matched only AFTER tuning the local learning rate down (5e-5);
naive FedAvg at 1e-3 *degraded* the model (oscillating 0.51–0.66) even
with IID data.
Mechanism: averaging two models that each took large independent steps
lands off the shared minimum. Small local steps keep clients close enough
that their average stays near it. We show the full sweep (1e-3 → 5e-5,
monotonic stabilization) rather than only the tuned result.

**C3. C uploads 16× less than A. Where's the catch?**
Short: the catch is edge compute (158 s of local training vs 0) and the
cost structure: C's traffic scales with model size and round count, A/B's
with data volume.
Consequence shown in F6: the fusion model (3.5× more trainable parameters)
costs 3.5× more per round (1.41 MB vs 400 KB) — federated bandwidth is a
function of the model, not the data. With little data, A/B can be cheaper;
with much data, C always wins.

**C4. Why hand-rolled FedAvg over MQTT instead of the Flower framework?**
Short: comparability. A/B/C share one transport, so bandwidth differences
are attributable to the strategy, not to protocol overhead differences
(Flower uses gRPC).
FedAvg itself is ~10 lines (a weighted mean); the framework adds
orchestration we already had via MQTT. Flower is acknowledged as the
production path; our scaffolds for it were removed for honesty (unwired
code).

---

## D. Federated learning deep-dives

**D1. Walk me through one FedAvg round.**
Server broadcasts global weights → each client overwrites its local model
with them → each trains E epochs on ITS OWN data → each sends weights + its
sample count → server computes w_global = Σ (n_k/n)·w_k → broadcast; repeat.
Data never moves; the model commutes.

**D2. Your two nodes see similar data. What if they were in different
rooms (different distributions)?**
Short: that is the non-IID setting; vanilla FedAvg degrades under it
(client drift: each copy is pulled toward its room; the average can be
worse than either).
Status: our splits are near-IID by design (isolating the placement
variable); the deliberate non-IID experiment (skewed class split per node,
show drift, optionally FedProx as the fix) is a planned rung.
Worth noting: mild heterogeneity is the POINT of federation — averaging
transfers knowledge between rooms without data sharing (node A improves on
traffic it never heard).

**D3. Is federated learning actually private? Weights leave the device.**
Short honest answer: it removes the *raw-data* exposure, which is the
biggest and most legible risk; it is not information-theoretically private.
Known attacks (gradient inversion, membership inference) can extract some
training-data information from weight updates. Standard hardening exists —
secure aggregation, differential privacy — and is out of scope but
acknowledged. Our claim is precisely "raw audio never leaves the device,"
not "zero information leaves the device."

**D4. Why a FRESH optimizer per round on the clients?**
Short: standard FedAvg carries no optimizer state across rounds; reusing
Adam momentum after set_weights() applies stale momentum to fresh weights
and corrupts the first steps. We hit this as a real bug (accuracy collapse)
before fixing it — it's documented.

---

## E. Fusion

**E1. Why did naive fusion perform WORSE than vision alone?**
Short: modality gradient starvation — the pre-trained eye explains the
labels immediately, so the from-scratch ear receives almost no learning
signal, stays uninformative, and the head learns to ignore it. Fusion then
adds noise, not information.
Fix (and the finding): warm-start the audio branch from the trained
audio-only model, THEN train with 50% modality dropout → 0.847 clean,
beating both baselines.

**E2. How do you know the model actually uses audio now?**
Three probes: blackout accuracy 0.551 (vs 0.204 chance-level for
vision-only); predictions CHANGE when only the audio input is swapped;
audio-branch feature variance is healthy rather than near-constant. Before
the fix, all three probes failed (constant class, 0/20 sensitivity, flat
features).

**E3. Doesn't the blackout test favor you? Real fog isn't pure black.**
Fair. Pure black is the extreme end of a spectrum; it gives a clean,
reproducible worst case. If pushed we would add intermediate degradations
(blur, noise, low contrast, partial occlusion) and show the degradation
curve rather than two endpoints. Expectation: monotonic decline for
vision-only, flat-ish floor for the fused model.

**E4. Why ship the feature vector and not the frame in FB? Isn't the FV
also derived from the frame?**
Short: 1.1 KB vs ~30–50 KB, and the FV is a lossy 576-number summary that
cannot be inverted into a recognizable image — so bandwidth AND privacy
both improve. The frozen backbone makes it safe: its output never changes,
so edge and server always agree (we verify 100% prediction agreement
between the FV-model and the image-model).
Honest nuance: an FV still leaks *something* (it's classifiable
information); the claim is "no reconstructable image leaves," not "no
information leaves."

**E5. Why is the vision branch frozen? Why not fine-tune it?**
Short: three reasons — edge devices can't afford to train a 940K-param
image backbone; frozen means the FV wire format stays valid (a fine-tuned
backbone on the server would diverge from the edge's copy); and ImageNet
features are already strong for our classes. Trade-off acknowledged:
fine-tuning could add a few points of vision accuracy at the cost of the
entire FB/FC placement design.

**E6. VGGSound labels only the SOUND. Is the object always visible in your
frames?**
Not always — a "dog barking" clip can show a living room. That's label
noise on the VISION side, analogous to YOLO's 20% on the audio side. The
vision-only baseline's 0.806 shows visibility is high in practice (these
are videos OF the thing, a selection bias we note). If pushed: filter pairs
by running YOLO on the frames and keeping agreement cases — at the cost of
dataset size and a new selection bias.

---

## F. Hardware & engineering

**F1. What did the real-Pi validation actually prove?**
That the pipeline is not a simulation artifact: live capture → YOLO
(~430 ms/frame on Pi 5 CPU) → buffering → MQTT over real WiFi → the
UNCHANGED science server, with sender and receiver byte counts matching
exactly. The server cannot tell a real Pi from a simulated one — which is
the design working as intended (strategy pattern; only the sensor
front-end differs).

**F2. Anything go wrong with hardware?** (good story, tell it)
The webcam's built-in mic produced a constant full-scale signal — detected
not by driver errors but by inspecting the data (RMS == max ⇒ constant).
Replaced with a separate USB mic; the lesson "validate sensors by looking
at their data" is written into the thesis. Also: YOLO's first inference
costs ~4 s (warmup) — measured and excluded by design.

**F3. Can a Pi actually train the fusion model (Strategy FC on hardware)?**
Laptop reference points exist (backbone 3.9 ms/frame; local epoch ~7.5 s);
the Pi measurement is the remaining F7 rung. Expectation: ~50–100× slower
backbone (~200–400 ms/frame — matches YOLO's Pi slowdown factor) and
minutes per local epoch — feasible at our batch cadence, to be confirmed
by measurement, not assumption.

**F4. What breaks at 10 or 100 nodes?**
MQTT broker and FedAvg both scale fine (mean over N). Real issues:
stragglers (the round waits for the slowest client — mitigated by
timeouts/partial aggregation), server round-time for A/B grows with pooled
data (measured: linear), and non-IID drift grows with heterogeneity.
None are architecture-breaking; all are known FL engineering.

---

## G. Scope & framing

**G1. Your title says audio-VISUAL fusion, but the strategy comparison used
an audio-only model. Explain.**
Two answers, both true: (1) the system was always audio-visual — vision
LABELS audio (cross-modal supervision) even when the trained model is
audio-only; (2) the full audio+vision fusion model is now also trained,
evaluated (0.847/0.551), and run under the strategies (FB/FC). The
comparison methodology was deliberately established on the simpler model
first — one variable at a time.

**G2. What is the single main contribution?**
A measured, end-to-end comparison of training placement (centralized /
hybrid / federated) for a self-labeling audio-visual edge system — with
honest secondary findings: compression erases most of hybrid's theoretical
advantage; naive FedAvg destabilizes converged models (LR sensitivity);
naive multimodal fusion silently ignores the weak modality and a
warm-start + modality-dropout recipe fixes it; federated bandwidth scales
with model size, not data volume.

**G3. What would you do with three more months?**
Priority order: model-freshness experiment (adaptation speed to unseen
classes — the sharpest A-vs-C differentiator); non-IID + FedProx; fusion
strategies on real Pis (F7 + live FC); 10-fold CV and a larger fusion test
set; intermediate-degradation curves for the fog test.

---

*Tip for the defense: when a number is challenged, the winning move is the
provenance chain — every figure in RESULTS.md traces to a JSON in results/
with the exact command in invocation.txt, and byte counts are double-entry
(sender and receiver logged independently and matching).*
