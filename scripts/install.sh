#!/usr/bin/env bash
# Snippy installer.
#
#   ./install.sh              install, ask about spoken triggers, start in the background
#   ./install.sh --asr        install with speech recognition and start
#   ./install.sh --no-asr     install without speech recognition
#   ./install.sh --no-systemd install only, no service
#   ./install.sh --uninstall  stop the service and remove the data directory
#
# The goal is that a fresh clone becomes a running background process in three
# commands: clone, install, and then forget about it.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

BOLD=$'\033[1m'; DIM=$'\033[2m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
ok()   { printf '%s  ok%s    %s\n' "$GREEN" "$OFF" "$1"; }
warn() { printf '%s  warn%s  %s\n' "$YELLOW" "$OFF" "$1"; }
die()  { printf '%s  fail%s  %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }
info() { printf '%s        %s%s\n' "$DIM" "$1" "$OFF"; }
head_() { printf '\n%s%s%s\n' "$BOLD" "$1" "$OFF"; }

WANT_ASR="ask"
USE_SYSTEMD="yes"
UNINSTALL="no"

for arg in "$@"; do
  case "$arg" in
    --asr) WANT_ASR="yes" ;;
    --no-asr) WANT_ASR="no" ;;
    --no-systemd) USE_SYSTEMD="no" ;;
    --uninstall) UNINSTALL="yes" ;;
    -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

UNIT_NAME="snippy"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

# ---------------------------------------------------------------- uninstall --
if [ "$UNINSTALL" = "yes" ]; then
  head_ "Removing Snippy"
  if command -v systemctl >/dev/null 2>&1; then
    systemctl --user stop "$UNIT_NAME" 2>/dev/null || true
    systemctl --user disable "$UNIT_NAME" 2>/dev/null || true
    rm -f "${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/$UNIT_NAME.service"
    systemctl --user daemon-reload 2>/dev/null || true
    ok "service removed"
  fi
  if [ -f "$ROOT/snippy.pid" ]; then
    kill "$(cat "$ROOT/snippy.pid")" 2>/dev/null || true
    rm -f "$ROOT/snippy.pid"
  fi
  if [ -d "$ROOT/data" ]; then
    rm -rf "$ROOT/data"
    ok "data directory removed"
  fi
  [ -d "$VENV" ] && rm -rf "$VENV" && ok "virtualenv removed"
  exit 0
fi

# ------------------------------------------------------------------ python ---
head_ "Checking Python"
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo 0.0)"
    major="${version%%.*}"; minor="${version##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done
[ -n "$PYTHON_BIN" ] || die "Python 3.11 or newer is required. On Debian/Ubuntu: apt install python3.12 python3.12-venv"
PY_MINOR="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "python $PY_MINOR ($PYTHON_BIN)"

# ------------------------------------------------------------------ ffmpeg ---
head_ "Checking ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)"
else
  warn "ffmpeg is missing. Snippy needs it to encode and decode clips."
  info "Debian/Ubuntu: sudo apt install ffmpeg"
  info "macOS:         brew install ffmpeg"
  die "install ffmpeg, then run ./install.sh again"
fi

# -------------------------------------------------------------------- venv ---
head_ "Setting up the environment"
if [ ! -d "$VENV" ]; then
  "$PYTHON_BIN" -m venv "$VENV" || die "could not create a virtualenv (is python3-venv installed?)"
  ok "virtualenv created"
else
  ok "virtualenv already present"
fi

"$PY" -m pip install --quiet --upgrade pip setuptools wheel
ok "pip up to date"

REQ="$ROOT/requirements.txt"
if [ "$WANT_ASR" = "yes" ]; then
  REQ="$ROOT/requirements-asr.txt"
fi

# Python 3.13 removed the stdlib `audioop` module that discord.py imports.
if [ "$("$PY" -c 'import sys; print(sys.version_info >= (3, 13))')" = "True" ]; then
  info "python 3.13+ detected, pinning the audioop-lts shim"
  "$PY" -m pip install --quiet audioop-lts || warn "could not install audioop-lts; voice may not work"
fi

"$PY" -m pip install --quiet -r "$REQ" || die "dependency install failed"
if [ "$WANT_ASR" = "yes" ]; then
  ok "dependencies installed (with speech recognition)"
else
  ok "dependencies installed"
fi

# --------------------------------------------------------------------- env ---
if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  warn "created .env from the template"
  info "add your bot token to $ROOT/.env, then re-run ./install.sh"
  exit 1
fi
# shellcheck disable=SC1091
set -a; . "$ROOT/.env"; set +a
if [ -z "${DISCORD_TOKEN:-}" ] || [ "$DISCORD_TOKEN" = "your-bot-token-here" ]; then
  die "DISCORD_TOKEN is not set in .env"
fi
if [ "$WANT_ASR" = "ask" ]; then
  head_ "Spoken triggers"
  info "say \"hey snippy, clip this\" in voice and Snippy will cut it"
  printf '  Enable speech recognition? it adds a ~75MB model and some CPU [y/N] '
  read -r answer </dev/tty || answer="n"
  case "$answer" in
    y|Y|yes|YES) WANT_ASR="yes" ;;
    *) WANT_ASR="no" ;;
  esac
  if [ "$WANT_ASR" = "yes" ]; then
    "$PY" -m pip install --quiet -r "$ROOT/requirements-asr.txt" || die "speech recognition install failed"
    ok "speech recognition installed"
  fi
fi

# ------------------------------------------------------------------- check ---
head_ "Verifying the install"
mkdir -p "$ROOT/data/archive" "$ROOT/data/clips"
ASR_FLAG=""
[ "$WANT_ASR" = "yes" ] && ASR_FLAG="--asr"
CHECK_OUT="$("$PY" -m snippy --check $ASR_FLAG 2>&1)" || true
printf '%s\n' "$CHECK_OUT" | sed 's/^/  /'
if printf '%s' "$CHECK_OUT" | grep -q "problem(s) found"; then
  die "the install is not ready yet; see the failures above"
fi

# ----------------------------------------------------------------- service ---
if [ "$USE_SYSTEMD" = "no" ]; then
  head_ "Done (no service)"
  info "start it in the background with: ./scripts/run-daemon.sh"
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1; then
  warn "systemd is not available on this machine"
  info "falling back to a plain background process"
  "$ROOT/scripts/run-daemon.sh"
  exit 0
fi

head_ "Starting the background service"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
sed -e "s|@ROOT@|$ROOT|g" -e "s|@PYTHON@|$PY|g" \
  "$ROOT/systemd/snippy.service" > "$UNIT_DIR/snippy.service"

systemctl --user daemon-reload
systemctl --user enable --now snippy.service

# This is the part that makes the bot survive logging out and closing the SSH
# session. Without linger, systemd tears user services down on logout and
# Snippy dies the moment you disconnect.
if command -v loginctl >/dev/null 2>&1; then
  if loginctl enable-linger "$USER" 2>/dev/null; then
    ok "linger enabled: Snippy keeps running after you log out"
  else
    warn "could not enable linger; Snippy may stop when you log out"
    info "run: sudo loginctl enable-linger $USER"
  fi
fi

sleep 2
if systemctl --user is-active --quiet snippy.service; then
  ok "service is running"
else
  warn "service did not start cleanly; recent log:"
  journalctl --user -u snippy.service -n 20 --no-pager 2>/dev/null | sed 's/^/    /'
  exit 1
fi

head_ "${BOLD}Snippy is running.${OFF}"
info "status:   systemctl --user status snippy"
info "logs:     journalctl --user -u snippy -f"
info "restart:  systemctl --user restart snippy"
info "stop:     systemctl --user stop snippy"
info "settings: /snippy settings"
