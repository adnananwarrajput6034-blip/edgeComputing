# Pi Captures

Scratch area for images and audio pulled off the Raspberry Pi 5 edge node (`pi1`)
during hardware bring-up and sensor testing.

Contents are **not tracked in git** — they are raw sensor dumps, regenerable at
any time by re-running a capture against the Pi.

```
images/   webcam frames (JPEG, /dev/video0, Sonix 1080P FHD)
audio/    microphone recordings (WAV 16 kHz mono, plughw:2,0, Generalplus USB)
```

## Hardware

| | |
|---|---|
| Host | `pi1` (alias in `~/.ssh/config`, key auth, user `adi6034`) |
| Camera | `/dev/video0` — MJPG/YUYV at 1920x1080, 1280x720, 640x480 |
| Microphone | ALSA card 2, `plughw:2,0` |

The webcam exposes its own onboard mic as ALSA card 3. Always address the real
USB mic explicitly as `plughw:2,0` (or by name, `plughw:Device,0`) — card
numbering can swap across reboots depending on USB enumeration order.

## Capturing

```bash
scripts/pull_from_pi.sh          # one frame + 3 s of audio
scripts/pull_from_pi.sh -d 10    # 10 s of audio
```

Files are timestamped, so repeated runs accumulate rather than overwrite.
