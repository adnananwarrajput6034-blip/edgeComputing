"""
Live-vocabulary helpers shared by the three Pi clients
======================================================

The replay experiments (strategy_{a,b,c}_client.py) draw samples from
UrbanSound8K, so their label space is fixed at import time. The demo track
cannot work that way: the vocabulary is whatever objects someone puts in
front of the camera, decided at runtime.

This module holds the pieces all three live clients need:

    MicRecorder        500ms mic chunks, native rate -> 16 kHz int16
    enroll()           capture loop: YOLO labels live audio, live readout
    save/load_dataset  the enrolled samples as a portable .npz
    build_live_model   AudioCNN head resized to the enrolled vocabulary,
                       conv stack warm-started from the UrbanSound8K model

Enrollment happens ONCE (scripts run pi_enroll.py); the same frozen dataset
is then replayed through A, B and C. Capturing separately per strategy
would give each a different sample set and make the bandwidth comparison
meaningless — same data, different placement is the whole point.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGET_SR = 16000
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)  # 8000

WARM_START = REPO_ROOT / "models/audio_cnn_urbansound8k.keras"


# --------------------------------------------------------------------------
# microphone
# --------------------------------------------------------------------------

class MicRecorder:
    """500ms chunks at the mic's native rate, resampled to 16 kHz int16.

    USB mics rarely support 16 kHz directly, so we record at whatever the
    device offers and resample. Identical to the recorder in
    strategy_a_pi_client.py — kept here so all live clients share one copy.
    """

    def __init__(self, device: int | None):
        import sounddevice as sd
        self._sd = sd
        self.device = device
        info = sd.query_devices(device, "input")
        self.native_sr = int(info["default_samplerate"])
        self.name = info["name"]

    def start_chunk(self):
        frames = int(self.native_sr * CHUNK_SECONDS)
        return self._sd.rec(frames, samplerate=self.native_sr, channels=1,
                            dtype="float32", device=self.device)

    def finish(self, recording) -> np.ndarray:
        import librosa
        self._sd.wait()
        audio = recording[:, 0]
        if self.native_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=self.native_sr,
                                     target_sr=TARGET_SR)
        if len(audio) < CHUNK_SAMPLES:
            audio = np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))
        audio = audio[:CHUNK_SAMPLES]
        return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)


# --------------------------------------------------------------------------
# dataset io
# --------------------------------------------------------------------------

def save_dataset(path: Path, audio: np.ndarray, labels: list[str],
                 classes: list[str], test_fraction: float = 0.25,
                 seed: int = 0) -> dict:
    """Write enrolled samples to .npz with a stratified train/test split.

    The split is stratified per class so a class enrolled with few samples
    still appears in both halves; without that a small class can land
    entirely in train and the reported accuracy silently ignores it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    y = np.array([classes.index(l) for l in labels], dtype=np.int64)
    rng = np.random.default_rng(seed)

    test_idx: list[int] = []
    for c in range(len(classes)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction))) if len(idx) > 1 else 0
        test_idx.extend(idx[:n_test].tolist())

    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True

    np.savez_compressed(
        path,
        audio_train=audio[~test_mask], y_train=y[~test_mask],
        audio_test=audio[test_mask], y_test=y[test_mask],
        classes=np.array(classes, dtype=object),
    )
    return {
        "path": str(path),
        "classes": classes,
        "n_train": int((~test_mask).sum()),
        "n_test": int(test_mask.sum()),
        "per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)},
    }


def load_dataset(path: Path) -> dict:
    """Read an enrolled dataset back. Mirrors save_dataset."""
    z = np.load(Path(path), allow_pickle=True)
    return {
        "audio_train": z["audio_train"],
        "y_train": z["y_train"],
        "audio_test": z["audio_test"],
        "y_test": z["y_test"],
        "classes": [str(c) for c in z["classes"]],
    }


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_live_model(num_classes: int, learning_rate: float = 1e-3,
                     warm_start: Path | None = WARM_START, verbose: bool = True):
    """AudioCNN sized to the live vocabulary.

    The UrbanSound8K model has a 10-way head that cannot classify live
    classes, but its conv stack learned general spectrogram features
    (onsets, harmonic structure) that transfer. So we copy every layer
    except the final Dense and attach a fresh head of the right width.

    Falls back to random init when the warm-start file is absent.
    """
    import tensorflow as tf
    from src.server.training.models.audio_cnn import AudioCNN

    model = AudioCNN(num_classes=num_classes)
    warm_started = False

    if warm_start is not None and Path(warm_start).exists():
        src = tf.keras.models.load_model(warm_start)
        # Same architecture except the last layer; copy weights positionally
        # and stop before the head so a 10-way -> N-way change is safe.
        copied = 0
        for dst_layer, src_layer in zip(model.layers[:-1], src.layers[:-1]):
            if dst_layer.get_config().get("name", "") and src_layer.weights:
                try:
                    dst_layer.set_weights(src_layer.get_weights())
                    copied += 1
                except ValueError:
                    pass  # shape mismatch: leave randomly initialised
        warm_started = copied > 0
        if verbose:
            print(f"warm-started {copied} layers from {Path(warm_start).name} "
                  f"(new {num_classes}-way head)", flush=True)
    elif verbose:
        print("no warm-start model found — training from scratch", flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, warm_started


def predict_with_unknown(model, X: np.ndarray, classes: list[str],
                         threshold: float = 0.0):
    """Argmax prediction with an explicit `unknown` reject option.

    A softmax head is closed-set: every input is forced into one of the
    trained classes, however unlike them it is. Anything whose top
    probability falls below `threshold` is reported as "unknown" instead,
    so out-of-vocabulary sounds are refused rather than silently
    misfiled. threshold=0.0 disables the reject and restores plain argmax.
    """
    probs = model.predict(X, verbose=0)
    top = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out = [classes[i] if c >= threshold else "unknown"
           for i, c in zip(top, conf)]
    return out, conf, probs


# --------------------------------------------------------------------------
# enrollment
# --------------------------------------------------------------------------

def enroll(camera_index: int, audio_device: int | None, yolo_model: str,
           confidence: float, min_samples: int, duration_min: float | None,
           tick_ms: int, stop_flag: dict, evidence_dir: Path | None = None,
           ignore: tuple[str, ...] = ()):
    """Capture loop: YOLO names whatever is in frame, the mic supplies the
    audio, and samples accumulate under that name.

    Returns (audio_array, labels, stats). The caller decides the vocabulary
    afterwards by dropping classes below `min_samples` — a passer-by
    detected twice should not become a class.

    `ignore` drops labels from consideration entirely. The operator stands
    in front of the camera to run the thing, and a person at ~0.9 outscores
    a held-up object at ~0.6 on almost every tick — so without this the
    enrolled vocabulary is just "person". Ignoring it lets the object win
    instead of forcing you out of frame.

    Prints a live readout every tick so the operator can see what YOLO is
    detecting BEFORE relying on it; enrolling blind is how you discover at
    training time that the object was never recognised.
    """
    import cv2
    from src.edge.processing.vision import VisionProcessor

    ignore_set = {s.strip().lower() for s in ignore if s.strip()}
    if ignore_set:
        print(f"[enroll] ignoring labels: {sorted(ignore_set)}", flush=True)

    print(f"[enroll] opening camera {camera_index}...", flush=True)
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise RuntimeError(f"cannot open camera {camera_index}")

    mic = MicRecorder(audio_device)
    print(f"[enroll] mic: {mic.name} @ {mic.native_sr} Hz -> {TARGET_SR} Hz",
          flush=True)

    print(f"[enroll] loading YOLO ({yolo_model})...", flush=True)
    vision = VisionProcessor(model_path=yolo_model,
                             confidence_threshold=confidence)
    ok, warm = cam.read()
    if ok:
        vision.detect(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB))
    print("[enroll] YOLO warmed up", flush=True)

    samples: list[np.ndarray] = []
    labels: list[str] = []
    counts: Counter = Counter()
    yolo_ms: list[float] = []
    ticks = 0

    t_start = time.perf_counter()
    deadline = (t_start + duration_min * 60) if duration_min else None

    print("\n[enroll] SHOW AN OBJECT TO THE CAMERA AND MAKE ITS SOUND.")
    print("[enroll] Ctrl-C when every class has enough samples.\n", flush=True)

    while not stop_flag.get("flag"):
        if deadline and time.perf_counter() > deadline:
            break
        tick_t0 = time.perf_counter()
        ticks += 1

        rec = mic.start_chunk()        # 500ms window opens
        ok, frame_bgr = cam.read()     # frame lands inside that window
        audio = mic.finish(rec)
        if not ok:
            print("[enroll] WARN camera read failed", flush=True)
            continue

        y0 = time.perf_counter()
        detections = vision.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if ignore_set:
            detections = [d for d in detections
                          if d["class"].lower() not in ignore_set]
        picked = vision.get_dominant_label(detections)
        label, conf = picked if picked else (None, 0.0)
        yolo_ms.append((time.perf_counter() - y0) * 1000)

        if label is None:
            print(f"\r[enroll] tick {ticks}: (nothing detected)"
                  f"{' ' * 30}", end="", flush=True)
        else:
            samples.append(audio)
            labels.append(label)
            counts[label] += 1
            if evidence_dir is not None:
                import soundfile as sf
                stem = f"tick{ticks:04d}_{label.replace(' ', '_')}"
                cv2.imwrite(str(evidence_dir / f"{stem}.jpg"), frame_bgr)
                sf.write(str(evidence_dir / f"{stem}.wav"), audio, TARGET_SR)
            ready = [f"{c}:{n}" for c, n in counts.most_common()]
            print(f"\r[enroll] tick {ticks}: {label} ({conf:.2f})  |  "
                  f"{'  '.join(ready)}{' ' * 10}", end="", flush=True)

        delay = tick_ms / 1000.0 - (time.perf_counter() - tick_t0)
        if delay > 0:
            time.sleep(delay)

    cam.release()
    print()

    audio_arr = (np.stack(samples) if samples
                 else np.zeros((0, CHUNK_SAMPLES), dtype=np.int16))
    stats = {
        "ticks": ticks,
        "labeled": len(labels),
        "detection_rate": round(len(labels) / max(ticks, 1), 3),
        "yolo_ms_mean": round(float(np.mean(yolo_ms)), 1) if yolo_ms else None,
        "raw_counts": dict(counts),
        "wall_seconds": round(time.perf_counter() - t_start, 1),
    }
    return audio_arr, labels, stats
