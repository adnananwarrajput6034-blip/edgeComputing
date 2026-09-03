#!/usr/bin/env bash
# Countdown, take ONE snapshot on the Pi webcam, pull it here, and report focus.
#
#   scripts/pi_focus_check.sh           # 5 s countdown, then snap
#   scripts/pi_focus_check.sh -t 10     # 10 s countdown
#   scripts/pi_focus_check.sh -r 640x480
#
# Use the countdown to aim the camera / adjust the lens, then read the
# sharpness score printed with the saved image.
#
# Rough guide (textured scene, subject ~1 m away):
#   < 20    badly out of focus or obstructed lens
#   20-100  soft
#   > 100   sharp
#
# A blank wall scores low however well focused it is — aim at fine detail
# (text, a keyboard, a book spine).
set -euo pipefail

HOST="${PI_HOST:-pi1}"
WAIT=5
RES="1280x720"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--wait)       WAIT="$2"; shift 2 ;;
    -r|--resolution) RES="$2"; shift 2 ;;
    -h|--help)       sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/pi_captures/images"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "error: cannot reach '$HOST' over ssh" >&2; exit 1
fi

echo "Aim the camera — snapping in ${WAIT}s"
for ((i=WAIT; i>0; i--)); do printf "\r  %2ds..." "$i"; sleep 1; done
printf "\r  snap!   \n"

W="${RES%x*}"; H="${RES#*x}"
ssh "$HOST" "python3 -u -c '
import cv2, sys
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*\"MJPG\"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, $W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, $H)
if not cap.isOpened(): sys.exit(\"cannot open /dev/video0\")
for _ in range(25): cap.read()          # settle auto-exposure
ok, f = cap.read()
cap.release()
if not ok: sys.exit(\"capture failed\")
cv2.imwrite(\"/tmp/_snap.jpg\", f, [cv2.IMWRITE_JPEG_QUALITY, 95])
g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
s = cv2.Laplacian(g, cv2.CV_64F).var()
clip = (g > 250).mean() * 100
v = \"SHARP\" if s > 100 else (\"soft\" if s > 20 else \"BLURRY\")
if clip > 10: v += \" + OVEREXPOSED\"
print(f\"sharpness={s:.1f}  brightness={g.mean():.0f}  clipped={clip:.1f}%  -> {v}\")
' 2>/dev/null"

scp -q "$HOST:/tmp/_snap.jpg" "$DEST/$STAMP.jpg"
ssh "$HOST" "rm -f /tmp/_snap.jpg"
echo "saved: data/pi_captures/images/$STAMP.jpg"
