#!/usr/bin/env bash
# Live detection monitor — what does YOLO actually see right now?
#
# Run this BEFORE enrolling. Hold an object, watch the label and confidence,
# and move it until the reading is stable and above ~0.7. Enrolling an object
# YOLO only detects intermittently wastes the whole capture: the class never
# reaches --min-samples and nothing gets saved.
#
#   scripts/pi_detect_check.sh          # 30 seconds
#   scripts/pi_detect_check.sh -t 60
set -euo pipefail
HOST="${PI_HOST:-pi1}"
SECS=30
[[ "${1:-}" == "-t" || "${1:-}" == "--time" ]] && SECS="$2"

ssh -t "$HOST" "cd ~/thesis/edge/edgeComputing && .venv/bin/python -u -c \"
import cv2, time
from ultralytics import YOLO
m = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
for _ in range(20): cap.read()
print('hold an object still and watch — aim for one label above 0.70')
print()
end = time.time() + $SECS
while time.time() < end:
    ok, f = cap.read()
    if not ok: continue
    r = m.predict(f, verbose=False, conf=0.25)[0]
    dets = sorted(r.boxes, key=lambda b: -float(b.conf))[:3]
    if not dets:
        print('  (nothing detected)'.ljust(70))
    else:
        top = '  '.join(f'{m.names[int(b.cls)]} {float(b.conf):.2f}' for b in dets)
        verdict = 'GOOD' if float(dets[0].conf) >= 0.70 else 'weak — move closer / better light'
        print(f'  {top:<50s} {verdict}')
cap.release()
\" 2>/dev/null"
