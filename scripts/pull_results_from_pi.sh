#!/usr/bin/env bash
# Pull the Pi's experiment results back to the laptop.
#
# Every live client writes its report on the Pi (pi_client_*.json, the
# enrolled dataset, evidence galleries). This brings them here so they can
# be shown and compared alongside the server-side results.
#
#   scripts/pull_results_from_pi.sh              # pull results/live/
#   scripts/pull_results_from_pi.sh -r results   # pull everything under results/
#
# Lands in data/pi_captures/results/<remote path>, outside results/ so a
# pulled report is never confused with one this laptop produced.
set -euo pipefail

HOST="${PI_HOST:-pi1}"
REMOTE_REPO="${PI_REPO:-thesis/edge/edgeComputing}"
SUBDIR="results/live"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--remote-dir) SUBDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$LOCAL/data/pi_captures/results"
mkdir -p "$DEST"

if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
  echo "error: cannot reach '$HOST' over ssh" >&2; exit 1
fi

if ! ssh "$HOST" "test -d '$REMOTE_REPO/$SUBDIR'"; then
  echo "error: $HOST:$REMOTE_REPO/$SUBDIR does not exist — has anything run yet?" >&2
  exit 1
fi

echo "pulling $HOST:$REMOTE_REPO/$SUBDIR -> data/pi_captures/results/"
rsync -az --itemize-changes "$HOST:$REMOTE_REPO/$SUBDIR/" "$DEST/$(basename "$SUBDIR")/"

echo
echo "--- pulled reports ---"
find "$DEST" -name "*.json" -newermt '-1 day' | sort | while read -r f; do
  echo "  ${f#"$LOCAL/"}"
done
