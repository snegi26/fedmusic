#!/usr/bin/env bash
# Start or stop a local simulation SuperLink that uses the current environment's
# packages instead of installing the app's dependencies for every run.
#
#   env/local-superlink.sh start   # then: flwr run . <connection with address ":local:">
#   env/local-superlink.sh stop
#
# Why: the SuperLink that `flwr run` starts on its own (Flower >= 1.39) runs
# `uv sync` into a fresh environment per run. peft depends on torch, so every run
# downloads a second torch, and that copy (from PyPI) would shadow ACE-Step's
# pinned CUDA build. `flwr run` reuses a SuperLink already answering on the local
# port, so starting this one first avoids both problems and works offline.
#
# State goes to $FLWR_HOME/local-superlink (default ~/.flwr).
set -euo pipefail

PORT="${FLWR_LOCAL_SUPERLINK_HTTP_API_PORT:-39091}"
STATE="${FLWR_HOME:-$HOME/.flwr}/local-superlink"
URL="http://127.0.0.1:$PORT/"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

up() { curl -s -o /dev/null --max-time 2 "$URL"; }

case "${1:-}" in
  start)
    if up; then echo "a SuperLink is already answering on $URL"; exit 0; fi
    mkdir -p "$STATE"
    nohup flower-superlink --insecure --simulation --isolation subprocess \
      --host 127.0.0.1 --port "$PORT" \
      --disable-runtime-dependency-installation \
      --database "$STATE/state.db" --log-file "$STATE/superlink.log" \
      > "$STATE/stdout.log" 2>&1 &
    echo $! > "$STATE/pid"
    for _ in $(seq 1 60); do
      up && { echo "local SuperLink ready on $URL (pid $(cat "$STATE/pid"))"; exit 0; }
      sleep 0.5
    done
    tail -20 "$STATE/stdout.log" >&2
    echo "local SuperLink did not start" >&2
    exit 1
    ;;
  stop)
    if [[ -f "$STATE/pid" ]]; then
      kill "$(cat "$STATE/pid")" 2>/dev/null || true
      rm -f "$STATE/pid"
    fi
    ;;
  *)
    sed -n '2,15p' "$0"
    exit 2
    ;;
esac
