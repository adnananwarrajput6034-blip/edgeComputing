"""
Preprocess harvested VGGSound pairs into a training cache (F2)
==============================================================

Input:  data/vggsound/pairs/{class}/{yt_id}.jpg + .wav
        (produced by spike_vggsound_pairs.py)

For each clip:
    frame:  jpg → 224×224 RGB uint8            (MobileNetV3 input size)
    audio:  10 s wav → N evenly-spaced 500 ms windows → (51,128) spectrograms
            (train clips get --train-windows windows; val/test get 1 center
             window — same convention as prepare_urbansound8k.py)

Every audio window of a clip pairs with that clip's single frame. To avoid
storing the same frame many times, frames are stored ONCE per clip in an
`images` array; each sample carries an index into it.

Split is BY CLIP (stratified per class, deterministic): windows of one clip
never straddle train/val/test — that would leak.

Output: data/vggsound/paired_cache.npz
    images            (n_clips, 224, 224, 3) uint8
    classes           (n_classes,) str
    per split s in {train,val,test}:
        img_idx_{s}   (N_s,) int32   — index into `images`
        X_aud_{s}     (N_s, 51, 128, 1) float32
        y_{s}         (N_s,) int32

Usage:
    .venv/bin/python scripts/prepare_vggsound_pairs.py
    .venv/bin/python scripts/prepare_vggsound_pairs.py --train-windows 4
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.edge.processing.stft import STFTProcessor  # noqa: E402

PAIRS_ROOT = REPO_ROOT / "data/vggsound/pairs"
CACHE_OUT = REPO_ROOT / "data/vggsound/paired_cache.npz"

SAMPLE_RATE = 16000
CHUNK = 8000  # 500 ms
IMG_SIZE = 224
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)  # train / val / test, by clip, per class
SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-windows", type=int, default=4,
                   help="500ms windows per TRAIN clip (val/test always 1 center)")
    p.add_argument("--out", default=str(CACHE_OUT))
    return p.parse_args()


def load_frame(jpg: Path) -> np.ndarray:
    with Image.open(jpg) as im:
        return np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                          dtype=np.uint8)


def audio_windows(wav: Path, n_windows: int) -> list[np.ndarray]:
    audio, sr = sf.read(wav, dtype="float32")
    if sr != SAMPLE_RATE:  # harvester writes 16k; guard anyway
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    n = len(audio)
    if n < CHUNK:
        pad = CHUNK - n
        return [np.pad(audio, (pad // 2, pad - pad // 2))]
    if n_windows == 1:
        s = (n - CHUNK) // 2
        return [audio[s:s + CHUNK]]
    starts = np.linspace(0, n - CHUNK, min(n_windows, n // CHUNK)).astype(int)
    return [audio[s:s + CHUNK] for s in starts]


def main() -> int:
    args = parse_args()
    if not PAIRS_ROOT.exists():
        print(f"ERROR: {PAIRS_ROOT} missing — run spike_vggsound_pairs.py first",
              file=sys.stderr)
        return 2

    classes = sorted(d.name for d in PAIRS_ROOT.iterdir() if d.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    # Collect complete pairs per class, deterministic order
    clips: list[tuple[str, Path, Path]] = []  # (class, jpg, wav)
    for c in classes:
        for jpg in sorted((PAIRS_ROOT / c).glob("*.jpg")):
            wav = jpg.with_suffix(".wav")
            if wav.exists():
                clips.append((c, jpg, wav))
    print(f"Complete pairs found: {len(clips)}")
    per_class = {c: sum(1 for x in clips if x[0] == c) for c in classes}
    print(f"Per class: {per_class}")

    # Split BY CLIP, stratified per class
    rng = random.Random(SEED)
    split_of: dict[Path, str] = {}
    for c in classes:
        cls_clips = [x for x in clips if x[0] == c]
        rng.shuffle(cls_clips)
        n = len(cls_clips)
        n_tr = int(n * SPLIT_FRACTIONS[0])
        n_va = int(n * SPLIT_FRACTIONS[1])
        for i, (_, jpg, _) in enumerate(cls_clips):
            split_of[jpg] = ("train" if i < n_tr
                             else "val" if i < n_tr + n_va else "test")

    stft = STFTProcessor(sample_rate=SAMPLE_RATE, n_fft=512, hop_length=160,
                         n_mels=128, window_length=400, normalize=True)

    images: list[np.ndarray] = []
    out: dict[str, list] = {f"{k}_{s}": [] for k in ("img_idx", "X_aud", "y")
                            for s in ("train", "val", "test")}

    t0 = time.perf_counter()
    for i, (c, jpg, wav) in enumerate(clips, 1):
        split = split_of[jpg]
        img_idx = len(images)
        images.append(load_frame(jpg))
        n_win = args.train_windows if split == "train" else 1
        for chunk in audio_windows(wav, n_win):
            spec = stft.process(chunk).astype(np.float32)[..., np.newaxis]
            out[f"img_idx_{split}"].append(img_idx)
            out[f"X_aud_{split}"].append(spec)
            out[f"y_{split}"].append(cls_to_idx[c])
        if i % 100 == 0:
            print(f"  {i}/{len(clips)} clips ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    arrays: dict[str, np.ndarray] = {
        "images": np.stack(images),
        "classes": np.array(classes),
    }
    for s in ("train", "val", "test"):
        arrays[f"img_idx_{s}"] = np.asarray(out[f"img_idx_{s}"], dtype=np.int32)
        arrays[f"X_aud_{s}"] = (np.stack(out[f"X_aud_{s}"])
                                if out[f"X_aud_{s}"] else
                                np.zeros((0, 51, 128, 1), np.float32))
        arrays[f"y_{s}"] = np.asarray(out[f"y_{s}"], dtype=np.int32)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)

    print(f"\nSaved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    print(f"{'SPLIT':<7}{'samples':>9}{'clips':>8}")
    for s in ("train", "val", "test"):
        n_samples = len(arrays[f"y_{s}"])
        n_clips = len(set(arrays[f"img_idx_{s}"].tolist()))
        print(f"{s:<7}{n_samples:>9}{n_clips:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
