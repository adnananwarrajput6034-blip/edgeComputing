#!/usr/bin/env bash
# Push local code to the Pi edge node.
#
# The Pi runs the same repo, so the live clients must be in sync before a
# demo. Copies code only — never data/, results/ or .venv/, so the Pi keeps
# its own captures and its own (aarch64) virtualenv.
#
#   scripts/sync_to_pi.sh          # sync and show what changed
#   scripts/sync_to_pi.sh -n       # dry run: list changes, copy nothing
set -euo pipefail

HOST="${PI_HOST:-pi1}"
REMOTE="${PI_REPO:-thesis/edge/edgeComputing}"
DRY=""
[[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]] && DRY="--dry-run"

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "error: cannot reach '$HOST' over ssh" >&2; exit 1
fi

echo "syncing code -> $HOST:$REMOTE ${DRY:+(dry run)}"
rsync -az --itemize-changes $DRY \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.venv/' --exclude 'venv/' \
  --exclude 'data/' --exclude 'results/' --exclude 'models/' \
  --exclude 'logs/' --exclude 'tensorboard_logs/' --exclude '.git/' \
  "$LOCAL/src/" "$HOST:$REMOTE/src/"
rsync -az --itemize-changes $DRY \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$LOCAL/configs/" "$HOST:$REMOTE/configs/"
echo "done"
