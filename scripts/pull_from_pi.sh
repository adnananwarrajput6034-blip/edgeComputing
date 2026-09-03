#!/usr/bin/env bash
# Capture a webcam frame + microphone clip on the Pi edge node and pull them here.
#
#   scripts/pull_from_pi.sh              # 1 frame + 3 s audio
#   scripts/pull_from_pi.sh -d 10        # 10 s of audio
#   scripts/pull_from_pi.sh -r 640x480   # lower capture resolution
#   scripts/pull_from_pi.sh --no-audio   # frame only
#   scripts/pull_from_pi.sh --no-image   # audio only
set -euo pipefail

HOST="${PI_HOST:-pi1}"
VIDEO_DEV="${PI_VIDEO_DEV:-/dev/video0}"
AUDIO_DEV="${PI_AUDIO_DEV:-plughw:2,0}"
RES="1920x1080"
DURATION=3
WARMUP=30          # frames to discard so auto-exposure settles
DO_IMAGE=1
DO_AUDIO=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--duration)   DURATION="$2"; shift 2 ;;
    -r|--resolution) RES="$2"; shift 2 ;;
    -w|--warmup)     WARMUP="$2"; shift 2 ;;
    --no-audio)      DO_AUDIO=0; shift ;;
    --no-image)      DO_IMAGE=0; shift ;;
    -h|--help)       sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/pi_captures"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST/images" "$DEST/audio"

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "error: cannot reach '$HOST' over ssh" >&2
  exit 1
fi

if [[ $DO_IMAGE -eq 1 ]]; then
  echo "capturing frame ($RES, $WARMUP warmup frames)..."
  ssh "$HOST" "ffmpeg -hide_banner -loglevel error -f v4l2 -input_format mjpeg \
      -video_size $RES -i $VIDEO_DEV -vf 'select=gte(n\,$WARMUP)' \
      -frames:v 1 -y /tmp/_pull.jpg"
  scp -q "$HOST:/tmp/_pull.jpg" "$DEST/images/$STAMP.jpg"
  ssh "$HOST" "rm -f /tmp/_pull.jpg"
  echo "  -> data/pi_captures/images/$STAMP.jpg"
fi

if [[ $DO_AUDIO -eq 1 ]]; then
  echo "recording ${DURATION}s of audio (16 kHz mono, $AUDIO_DEV)..."
  ssh "$HOST" "arecord -D $AUDIO_DEV -f S16_LE -r 16000 -c 1 \
      -d $DURATION /tmp/_pull.wav 2>/dev/null"
  scp -q "$HOST:/tmp/_pull.wav" "$DEST/audio/$STAMP.wav"
  ssh "$HOST" "rm -f /tmp/_pull.wav"
  echo "  -> data/pi_captures/audio/$STAMP.wav"
fi
