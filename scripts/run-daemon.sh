#!/usr/bin/env bash
# Start Snippy as a detached background process, for hosts without systemd.
#
# Setsid detaches it from this terminal's session, so closing the SSH connection
# does not send SIGHUP and the process keeps running. This is the fallback path;
# install.sh prefers a systemd user service, which restarts Snippy if it dies.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
PIDFILE="$ROOT/snippy.pid"
LOGFILE="$ROOT/data/snippy.log"

[ -x "$PY" ] || { echo "no virtualenv at $ROOT/.venv; run ./install.sh first" >&2; exit 1; }
mkdir -p "$ROOT/data"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running as pid $(cat "$PIDFILE")"
  exit 0
fi

# setsid + nohup + redirected stdio: survives the terminal going away, and does
# not keep the SSH session open waiting on the process.
setsid nohup "$PY" -m snippy >>"$LOGFILE" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
disown 2>/dev/null || true

sleep 2
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Snippy is running in the background (pid $(cat "$PIDFILE"))."
  echo "  logs:   tail -f $LOGFILE"
  echo "  stop:   kill \$(cat $PIDFILE)"
else
  echo "Snippy failed to start. Recent output:" >&2
  tail -20 "$LOGFILE" >&2 || true
  rm -f "$PIDFILE"
  exit 1
fi
