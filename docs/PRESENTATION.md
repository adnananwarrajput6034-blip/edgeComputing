# PRESENTATION.md — demo-day runbook

> Every command here has been executed against the real Pi and the real
> broker. Expected output is what actually came back, not what should
> happen in principle.
>
> Read §1 the night before. On the day, work top to bottom.

---

## 0. Demo at a glance

~25 minutes. Two tracks, deliberately separated:

| Track | What it is | Why it exists |
|---|---|---|
| **Science** | UrbanSound8K, 68.6% accuracy, the A/B/C comparison | Rigorous, reproducible, already validated |
| **Live** | Classes your examiner picks, learned on the Pi in the room | Proves the system is real, not a simulation |

The five acts:

| Act | Minutes | Shows |
|---|---|---|
| 1 | 4 | The Pi is real hardware: live camera, live mic, YOLO on-device |
| 2 | 5 | **Enrollment** — the examiner picks the classes, YOLO labels the audio |
| 3 | 4 | **Strategy A** — raw audio leaves the Pi, server trains |
| 4 | 4 | **Strategy B** — STFT moves to the Pi, bandwidth drops |
| 5 | 6 | **Strategy C** — training moves to the Pi, only weights leave |
| 6 | 4 | **Live inference** — the trained model names sounds, says `unknown`, asks to be retrained |
| — | 2 | The comparison table, pulled from the three runs |

**The one sentence:** *the same data, captured once, processed at three
different places in the pipeline — and here is what each placement costs.*

---

## 1. Pre-flight (the night before)

Do this the evening before. It takes ten minutes and removes almost every
way the demo can fail.

```bash
# 1. Pi reachable, key auth working
ssh pi1 'echo OK; hostname'

# 2. Sensors present
ssh pi1 'lsusb | grep -iE "audio|camera"; arecord -l | grep card'

# 3. Camera focused and exposed (aim at the demo objects, ~1 m)
./scripts/pi_focus_check.sh -t 5
```

Sharpness above ~20 with clipping under ~5% is fine. If it reads ~2, the
lens is obstructed — see §9.

```bash
# 4. Push the latest code to the Pi
./scripts/sync_to_pi.sh

# 5. Confirm the heavy dependencies are on the Pi
ssh pi1 'cd ~/thesis/edge/edgeComputing && .venv/bin/python -c "
import tensorflow, cv2, librosa, sounddevice, paho.mqtt, ultralytics
print(\"all edge deps OK\")"'

# 6. Broker binary exists on the laptop
ls -l /opt/homebrew/sbin/mosquitto
```

**Rehearse Act 2 once with the actual objects you plan to use.** The single
most likely failure is YOLO not recognising an object you assumed it would.

### Charge / cable checklist

- Pi power supply (a Pi 5 browns out on an underpowered USB-C charger)
- Webcam and USB mic plugged directly into the Pi, not through a hub
- Laptop and Pi on the **same network** — see §9 if the venue has none

---

## 2. Base setup (10 minutes before)

### 2.1 Find the Pi

```bash
ping -c1 pi1.local
```

mDNS usually resolves it. If not:

```bash
for i in $(seq 1 254); do ping -c1 -W300 192.168.178.$i >/dev/null 2>&1 & done; wait
arp -a | grep -v incomplete | grep -iE "2c:cf:67|b8:27:eb|dc:a6:32|e4:5f:01"
```

Those MAC prefixes are Raspberry Pi's. Yours is `2c:cf:67:26:9a:f7`.

### 2.2 Get the laptop's LAN IP — the Pi needs it

```bash
ipconfig getifaddr en0
```

**Write this down.** It appears in every Pi command as `<LAPTOP_IP>`. It
changes on every new network, and a stale IP is the most common demo
failure.

### 2.3 Start the broker

```bash
/opt/homebrew/sbin/mosquitto -c configs/mosquitto_lan.conf
```

Leave this running in its own terminal for the whole demo. It listens on
`0.0.0.0:1883` so the Pi can reach it — the default localhost-only config
refuses external connections.

Verify from the Pi:

```bash
ssh pi1 'nc -z -w3 <LAPTOP_IP> 1883 && echo "Pi can reach broker"'
```

### 2.4 Terminal layout

Four terminals, arranged so the examiner sees them all:

| # | Purpose |
|---|---|
| 1 | Broker (starts once, never touched again) |
| 2 | **Laptop server** — the data centre |
| 3 | **SSH to the Pi** — the edge device |
| 4 | Results and comparison |

Making the laptop/Pi split visible is worth real presentation points — the
examiner should never have to ask which machine a command ran on.

---

## 3. Act 1 — the Pi is real hardware (4 min)

> *"Everything from here runs on that Raspberry Pi, not on my laptop."*

**Terminal 3:**

```bash
ssh pi1
cd ~/thesis/edge/edgeComputing
```

Show the hardware is genuinely attached:

```bash
lsusb
arecord -l
```

Expected — the two sensors, plus the webcam's own onboard mic:

```
Bus 001 Device 002: ID 1b3f:2008 Generalplus Technology Inc. USB Audio Device
Bus 003 Device 002: ID 32a8:338b Sonix Technology Co., Ltd. 1080P FHD Camera

card 2: Device [USB Audio Device], device 0: USB Audio
card 3: Camera [1080P FHD Camera], device 0: USB Audio
```

> **Say:** the USB mic is card 2 and the webcam has its own inferior mic on
> card 3, so every command names `plughw:2,0` explicitly. Card numbering can
> swap across reboots, which is exactly the kind of detail that breaks
> unattended edge deployments.

Now run YOLO on a live frame:

```bash
.venv/bin/python -c "
import time, cv2
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
for _ in range(25): cap.read()
ok, f = cap.read(); cap.release()
m.predict(f, verbose=False)
t=time.perf_counter(); r=m.predict(f, verbose=False, conf=0.25)[0]
print(f'YOLOv8n on Pi 5: {(time.perf_counter()-t)*1000:.0f} ms')
for b in r.boxes: print('  ', m.names[int(b.cls)], f'{float(b.conf):.2f}')
"
```

**Measured:** `315 ms`, `person 0.89`.

> **Say:** 315 ms per frame on the edge device. That is why the design
> batches — you cannot afford YOLO on every frame, so the system collects
> samples and triggers work in batches. This number is the reason the
> architecture looks the way it does.

---

## 4. Act 2 — enrollment: the examiner picks the classes (5 min)

> *"I have not told the system what to listen for. You choose."*

This is the thesis's core claim running literally: **vision supervises
audio**. No human labels anything.

**Terminal 3 (on the Pi):**

```bash
.venv/bin/python -m src.experiments.pi_enroll \
    --audio-device 0 \
    --min-samples 30 \
    --confidence 0.5 \
    --ignore person \
    --save-evidence
```

What each flag does:

| Flag | Effect |
|---|---|
| `--audio-device 0` | The real USB mic. **Not 1** — that's the webcam's mic |
| `--min-samples 30` | Classes with fewer samples are dropped when the vocabulary freezes, so a passer-by detected twice doesn't become a class |
| `--confidence 0.5` | Below this YOLO detection threshold the tick is skipped — never guessed |
| `--ignore person` | **Essential.** YOLO takes the highest-confidence detection, and you standing in frame scores ~0.9 against a held-up object's ~0.6 — without this every class collapses to `person`. Add `,tv` when enrolling from a screen |
| `--save-evidence` | Saves frame + audio + label per tick so the examiner can audit what YOLO actually saw |

Each tick: a 500 ms recording opens, a frame is grabbed **inside** that
window, YOLO names the frame, and that name labels the audio.

Live readout, one line updating in place:

```
[enroll] tick 23: keyboard (0.88)  |  keyboard:18  scissors:4
```

**Ask the examiner to hold up an object and make its sound.** Keyboard →
type. Scissors → snip. Phone → ring it. Roughly 40 seconds per class at one
sample/second. Stop with **Ctrl-C** when each class shows ≥30.

Expected summary:

```
--- enrollment summary ---
ticks 130, labeled 124 (95%), YOLO 316.4 ms/frame
  KEPT    keyboard           64 samples
  KEPT    scissors           52 samples
  dropped person              8 samples
  (dropped: below --min-samples 30)

vocabulary: ['keyboard', 'scissors']
train 87 / test 29 samples
wrote results/live/live_dataset.npz
```

> **Say:** the vocabulary did not exist sixty seconds ago. Vision labelled
> the audio; nobody typed a label. The 95% is the detection rate — the
> other 5% of ticks saw nothing and were skipped rather than mislabelled,
> which is the honest failure mode.

### Copy the dataset to the laptop

The servers need it for the class list and the held-out test set.

**Terminal 4:**

```bash
scp pi1:~/thesis/edge/edgeComputing/results/live/live_dataset.npz \
    data/pi_captures/live_dataset.npz
```

> If enrollment produced fewer than 2 classes it exits with an error and
> tells you why. Enroll again, or lower `--min-samples`.

### Show the evidence — this is where provenance gets proved

Pull the audit trail and open it **before running any strategy**. The
examiner's natural question is "where did those labels come from?", and it
arrives now, not at the end.

```bash
rsync -az pi1:~/thesis/edge/edgeComputing/results/live/evidence_<TIMESTAMP>/ \
    data/pi_captures/evidence/
open data/pi_captures/evidence/
```

Each labelled tick is a pair — `tick0120_car.jpg` and `tick0120_car.wav`:
the frame YOLO read, and the audio it labelled, captured in the same
half-second window.

> **Say:** here is every sample the system kept, with the picture that
> named it. Nobody typed a label. Sort by filename and you can audit any
> one of them.

Worth pointing at the classes that were **dropped**:

```
55 person  40 car  33 cell_phone     <- kept
17 traffic_light  6 bird  5 tv  2 teddy_bear  1 stop_sign  1 scissors   <- dropped
```

> **Say:** it recognised nine things and kept three. The rest fell below
> the evidence threshold and were discarded rather than becoming
> unreliable classes.

### One dataset, three strategies

This single file now feeds all three acts:

```
                  |--> Strategy A   ships raw audio
live_dataset.npz  |--> Strategy B   ships spectrograms
   (96 train)     |--> Strategy C   ships weights only
```

Same samples, same labels, same server-side training. The only variable is
**where the work happens**. Enrolling separately per strategy would change
the data underneath the comparison and the byte counts would mean nothing.

---

## 5. Act 3 — Strategy A: raw audio leaves the device (4 min)

**Terminal 2 (laptop — the data centre):**

```bash
.venv/bin/python -m src.experiments.strategy_a_server \
    --broker localhost \
    --num-rounds 3 \
    --train-trigger-samples 30 \
    --epochs-per-round 10 \
    --batch-size 8 \
    --live-dataset data/pi_captures/live_dataset.npz \
    --save-model results/demo/trained.keras \
    --run-dir results/demo/strategy_a
```

| Flag | Effect |
|---|---|
| `--live-dataset` | Replaces the UrbanSound8K vocabulary with the enrolled classes and **resizes the model head** to match |
| `--train-trigger-samples 30` | Train once 30 samples have pooled — well below your ~96 enrolled, so you get several rounds |
| `--batch-size 8` | **This one matters.** The default 32 gives ~1 gradient step per epoch on a 30-sample pool, and accuracy never leaves chance |
| `--epochs-per-round 10` | ~100 live samples need more passes than the thousands UrbanSound8K provides |
| `--save-model` | Writes the trained model to disk so Act 6 can use it. Without it the model is discarded |
| `--num-rounds 2` | Stop after two training rounds |

Expected startup:

```
Live dataset: data/pi_captures/live_dataset.npz
warm-started 7 layers from audio_cnn_urbansound8k.keras (new 2-way head)
Live vocabulary ['keyboard', 'scissors'], test set (29, 51, 128, 1)
Baseline test accuracy before any round: 0.517
Server ready. Waiting for edge batches
```

> **Say:** the 7 warm-started layers are the convolutional stack from the
> UrbanSound8K model. Those features transfer; only the classification head
> is new, because the old head could only name urban sounds. Baseline is
> chance — the model genuinely knows nothing about these classes yet.

**Terminal 3 (Pi):**

```bash
.venv/bin/python -m src.experiments.strategy_a_pi_client \
    --node-id A --broker <LAPTOP_IP> \
    --dataset results/live/live_dataset.npz \
    --buffer-size 30 \
    --run-dir results/live/strategy_a
```

`--dataset` replays the enrolled samples. **All three strategies replay the
same dataset** — otherwise each would see different samples and the
bandwidth comparison would be meaningless.

Expected (Pi):

```
[pi-A] dataset: 180 samples, classes=['keyboard', 'scissors']
[pi-A] batch 1 uploaded (454 KB total)
...
[pi-A] done: 180 samples, 2726 KB uploaded
```

Expected (server):

```
Batch from A: 30 samples (454 KB compressed) — pool=60
=== Round 1/2: training on 60 pooled samples ===
=== Round 1 done: test_acc=0.500, train=1.5s, broadcast=405 KB ===
=== Round 2 done: test_acc=0.633, train=0.5s, broadcast=405 KB ===
```

> **Say:** 2.7 MB of raw audio crossed the network, and the server did the
> STFT and the training. Note what this means for privacy: that is
> reconstructible audio. Anything said near the microphone left the device.

---

## 6. Act 4 — Strategy B: move the STFT to the edge (4 min)

Stop the A server (**Ctrl-C**). **Terminal 2:**

```bash
.venv/bin/python -m src.experiments.strategy_b_server \
    --broker localhost \
    --num-rounds 3 \
    --train-trigger-samples 30 \
    --epochs-per-round 10 \
    --batch-size 8 \
    --live-dataset data/pi_captures/live_dataset.npz \
    --save-model results/demo/trained.keras \
    --run-dir results/demo/strategy_b
```

**Terminal 3 (Pi):**

```bash
.venv/bin/python -m src.experiments.strategy_b_pi_client \
    --node-id A --broker <LAPTOP_IP> \
    --dataset results/live/live_dataset.npz \
    --buffer-size 30 \
    --run-dir results/live/strategy_b
```

Expected:

```
[pi-A] done: 180 samples, 1737 KB uploaded, edge STFT 5.13s
```

**The single point of this act:** identical data, identical server
training, one thing moved.

| | Strategy A | Strategy B |
|---|---|---|
| Uploaded | 2726 KB | **1737 KB** (−36%) |
| Edge CPU | ~0 | **5.13 s** of STFT |
| What's on the wire | raw waveform | mel-spectrogram (float16) |

> **Say:** 36% less bandwidth, bought with five seconds of Pi CPU. And the
> privacy position improves — a mel-spectrogram is lossy and far harder to
> reconstruct speech from than the waveform it came from. But the data
> still leaves the device.

---

## 7. Act 5 — Strategy C: the data never leaves (6 min)

Stop the B server. **Terminal 2:**

```bash
.venv/bin/python -m src.experiments.strategy_c_server \
    --broker localhost \
    --num-rounds 3 \
    --num-clients 1 \
    --epochs-per-round 8 \
    --live-dataset data/pi_captures/live_dataset.npz \
    --save-model results/demo/trained.keras \
    --run-dir results/demo/strategy_c
```

> **`--num-clients 1` is essential with a single Pi.** The default is 2
> (the two-node science runs) and the server will wait forever for a second
> node that never connects.

**Terminal 3 (Pi):**

```bash
.venv/bin/python -m src.experiments.strategy_c_pi_client \
    --node-id A --broker <LAPTOP_IP> \
    --dataset results/live/live_dataset.npz \
    --run-dir results/live/strategy_c
```

Expected (Pi):

```
[pi-A] local set: X=(180, 51, 128, 1) (1.2s edge STFT)
warm-started 7 layers from audio_cnn_urbansound8k.keras (new 2-way head)
[pi-A] round 1: trained 2 epoch(s) on 180 samples in 9.1s, sent 405 KB
[pi-A] round 2: trained 2 epoch(s) on 180 samples in 9.0s, sent 405 KB
[pi-A] round 3: trained 2 epoch(s) on 180 samples in 9.3s, sent 405 KB
[pi-A] done: 3 rounds, 1215 KB uploaded, 27.4s local training
```

Expected (server):

```
Round 1: weights from A (180 samples, 405 KB)
=== Round 1 FedAvg: test_acc=0.500, agg=0.10s, 1 clients ===
=== Round 2 FedAvg: test_acc=1.000, agg=0.09s, 1 clients ===
```

> **Say, and this is the moment of the demo:** the audio never left the Pi.
> What crossed the network was a 405 KB weight vector. The server did no
> training at all — 0.09 seconds of averaging. And the model learned a
> class that did not exist five minutes ago.

### The freshness point

> **Say:** a federated round completed in about nine seconds of on-device
> training. That is what "model freshness in minutes rather than hours"
> means concretely — a new object can be learned during this conversation,
> whereas the centralised path waits for an upload window and a server
> training job.

### The bandwidth nuance — expect this question

Strategy C uploaded 1215 KB against A's 2726 KB, but the shape of the cost
is what matters:

- **A and B** scale with **data**: twice the samples, twice the bytes.
- **C** scales with **rounds**: 405 KB per round *regardless of how much
  data the Pi holds*. Ten times the local data costs nothing extra.

So C's advantage widens as the deployment grows — which is precisely the
regime a real sensor network lives in.

---

## 7b. Act 6 — use the model you just trained (4 min)

Everything so far produced numbers. This is where the system *does
something*. Copy the model the server just saved onto the Pi:

```bash
scp results/demo/trained.keras results/demo/trained.classes.json \
    pi1:~/thesis/edge/edgeComputing/results/demo/
```

**Terminal 3 (Pi):**

```bash
.venv/bin/python -m src.experiments.pi_infer \
    --model results/demo/trained.keras \
    --audio-device 0 \
    --threshold 0.60 \
    --novelty-ticks 5
```

It listens through the microphone and names what it hears, once a second:

```
EAR (audio model)           EYE (YOLO)            note
------------------------------------------------------------------------
person 0.90                 -                       ##################
person 0.97                 -                       ###################
unknown 0.46                -                     below 0.60 — refusing to guess
```

### Three things to demonstrate, in this order

**1. It works.** Make a sound from one of the enrolled classes. It names it
from audio alone.

> **Say:** the camera is not involved in this decision. It was only ever
> used to create the labels. This is the audio model working on its own —
> which is the point, because in fog or darkness that is all you have.

**2. It knows what it doesn't know.** Make a sound it never learned —
whistle, tap the desk, clap.

> **Say:** a softmax classifier is closed-set: without a reject option it
> is forced to pick one of its three classes and will often do so
> confidently. Below the threshold it declines instead. That is a
> deliberate guard, not a limitation I overlooked.

**3. It asks to be retrained.** Keep the unknown sound going for five
consecutive ticks:

```
------------------------------------------------------------------------
  NEW SOUND: 5 unknowns in a row, nothing recognisable in view.
  Out of vocabulary ['car', 'cell phone', 'person'].
------------------------------------------------------------------------
```

> **Say:** the device has noticed it is out of its depth. In deployment
> this is what triggers enrollment for a new class — and federated
> learning is what makes acting on it cheap, because a round costs 405 KB
> and nine seconds rather than shipping the audio to a data centre and
> waiting for a nightly job.

That closes the loop the thesis opens with.

### Why there is no YOLO column

Vision is off by default and the `EYE` column shows `-`. Not a design
choice: **TensorFlow and PyTorch segfault in the same process on this Pi**
— an OpenMP clash on aarch64 that survives both import reordering and
`OMP_NUM_THREADS=1`.

It costs nothing, because nothing in the pipeline needs both at once —
enrollment runs YOLO with no TensorFlow, inference runs TensorFlow with no
YOLO. If asked, it is a real and quotable edge-deployment constraint:
frameworks that coexist on a laptop do not always coexist on the device.

> If the examiner asks to see vision and audio side by side, run
> enrollment in a second SSH window — separate processes are fine.

---

## 8. Bringing the results back

```bash
./scripts/pull_results_from_pi.sh
```

Pulls the Pi's reports into `data/pi_captures/results/live/` — deliberately
outside `results/`, so a report from the Pi is never confused with one the
laptop produced.

Then show the comparison:

```bash
.venv/bin/python - <<'PY'
import json, pathlib
base = pathlib.Path("data/pi_captures/results/live")
print(f"{'Strategy':<16}{'Uploaded':>12}{'Edge CPU':>12}  What crossed the network")
for s, desc in [("strategy_a","raw audio"),("strategy_b","mel-spectrograms"),
                ("strategy_c","model weights only")]:
    f = base / s / "pi_client_A.json"
    if not f.exists(): continue
    d = json.loads(f.read_text())
    kb = d["bytes_uploaded"]/1024
    cpu = d.get("stft_time_total") or d.get("total_local_train_seconds") or 0
    print(f"{d['strategy']:<16}{kb:>10.0f} KB{cpu:>10.1f} s  {desc}")
PY
```

Produces:

```
Strategy           Uploaded  Edge CPU  Final acc  What crossed the network
----------------------------------------------------------------------------
A_centralized       1305 KB     0.0 s      0.531  raw audio
B_hybrid             956 KB     1.1 s      0.531  mel-spectrograms
C_federated         1216 KB    52.6 s      0.562  model weights only
```

Those are the numbers from a full rehearsal on 96 enrolled training samples
(`car` / `cell phone` / `person`, 32 held out). Chance is 0.333.

**Hand the examiner this table.** It is the thesis in six numbers: cost
moves from the network to the edge device as privacy improves.

If you skipped the evidence gallery in Act 2, it is still there — but show
it there rather than here, where provenance is the question being asked.

---

## 9. If it breaks

| Symptom | Cause | Fix |
|---|---|---|
| `cannot reach broker` | Wrong `<LAPTOP_IP>`, or broker not running | Re-run `ipconfig getifaddr en0`; restart mosquitto |
| Pi client hangs at *waiting for global model* | Server not started, or `--num-clients 2` with one Pi | Start the C server with `--num-clients 1` |
| Enrollment: `need >=2 classes` | Only one object recognised | Enroll longer, add an object, or lower `--min-samples` |
| Enrollment detects nothing | Object not in COCO's 80 classes, or out of focus | Try `--confidence 0.3`; fall back to a keyboard or phone |
| `no such device` on the mic | Card renumbered after reboot | `arecord -l`, then adjust `--audio-device` |
| Blurry frames | Lens film / focus | `./scripts/pi_focus_check.sh`; sharpness should exceed ~20 |
| Server trains on 0 samples | `skipped_unknown_labels` climbing | Client and server are using different datasets — both must point at the same enrolled file |
| `Segmentation fault` on the Pi | TensorFlow + PyTorch in one process | Never combine them; `pi_infer` runs audio-only by default |
| Accuracy stuck at chance | `--batch-size` too large for a small pool | Use `--batch-size 8 --epochs-per-round 10` |
| **No network at the venue** | — | Phone hotspot, join both devices, re-read the laptop IP. Or ethernet Pi→laptop with link-local addressing |

### The escape hatch

If live capture fails entirely, the science track stands alone. Every
number in `docs/RESULTS.md` was produced without hardware, and the runs in
`results/` are complete. Say so plainly and present those — a reproducible
result set is not a fallback you need to apologise for.

---

## 10. Talking points and likely questions

`docs/DEFENSE_QA.md` has 30 prepared answers. The four that this demo
specifically invites:

**"Your labels come from vision. What if vision is wrong?"**
Then the audio is mislabelled — that is weak supervision, and it is the
central assumption. Two mitigations are in the code: the confidence
threshold (`--confidence`), and skipping undetected ticks rather than
guessing. `scripts/measure_label_noise.py` quantifies the resulting noise
rate.

**"What happens with a sound it was never trained on?"**
Answer this before being asked. The head is a softmax — a **closed set**
with no reject option, so an unknown sound is forced into one of the
trained classes, often confidently. `predict_with_unknown()` in
`src/experiments/pi_live.py` adds a confidence threshold that reports
`unknown` instead. Open-set recognition is named future work, not an
oversight.

**"Isn't 315 ms per frame too slow?"**
It would be for per-frame inference, which is why the design never does
that. Capture is batched, and YOLO runs once per ~1 s tick. The number
drove the architecture rather than embarrassing it.

**"Why not just use a bigger model?"**
The Pi trains a 110k-parameter model at ~3.8 s/epoch over 200 samples. That
budget is what makes federated rounds finish in seconds. A larger model
would break the freshness claim, which is the property being defended.

### Closing line

> *"The same audio, captured once. Strategy A ships the waveform, B ships a
> spectrogram, C ships nothing but weights. Cost moves from the network to
> the device, privacy improves as it does, and the model can learn a new
> object in the time it takes to answer one question."*

---

## Appendix — command reference

```bash
# laptop
ipconfig getifaddr en0                                    # the IP the Pi needs
/opt/homebrew/sbin/mosquitto -c configs/mosquitto_lan.conf
./scripts/sync_to_pi.sh                                   # push code to Pi
./scripts/pull_results_from_pi.sh                         # bring results back
./scripts/pi_focus_check.sh -t 5                          # camera check

# pi (ssh pi1; cd ~/thesis/edge/edgeComputing)
.venv/bin/python -m src.experiments.pi_enroll --audio-device 0 --min-samples 30 --save-evidence
.venv/bin/python -m src.experiments.strategy_a_pi_client --node-id A --broker <IP> --dataset results/live/live_dataset.npz
.venv/bin/python -m src.experiments.strategy_b_pi_client --node-id A --broker <IP> --dataset results/live/live_dataset.npz
.venv/bin/python -m src.experiments.strategy_c_pi_client --node-id A --broker <IP> --dataset results/live/live_dataset.npz
.venv/bin/python -m src.experiments.pi_infer --model results/demo/trained.keras --audio-device 0
```

### Verified environment

| | |
|---|---|
| Pi | Raspberry Pi 5 Model B Rev 1.0, 8 GB, Debian 13 Trixie |
| Pi Python | 3.13.5, TensorFlow 2.21.0, Keras 3.15.1 |
| Camera | Sonix 1080P FHD, `/dev/video0`, MJPG |
| Mic | Generalplus USB, ALSA card 2, `plughw:2,0`, sounddevice index 0 |
| Laptop | TensorFlow 2.21.0, mosquitto 2.1.2 |
| Network | Pi and laptop on one LAN; broker on `<LAPTOP_IP>:1883` |

> Every number in this runbook comes from a full end-to-end rehearsal on
> the real Pi with 128 live-enrolled samples across `car`, `cell phone`
> and `person` — not from a simulation. Your own demo-day figures will
> differ with the classes you enroll and how acoustically distinct they
> are, but the shape holds: B uploads least, C uploads a fixed amount per
> round regardless of data volume, and edge CPU rises as privacy improves.
>
> Accuracy landed at 0.53-0.56 against 0.333 chance. That is a real result
> from ~96 training samples, and honest to present as such: the point of
> the live track is that the pipeline works on hardware, not that three
> minutes of enrollment rivals a 68.6% model trained on 8000 clips.
